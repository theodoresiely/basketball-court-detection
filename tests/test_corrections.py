import importlib.util
import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest

from courtdet.data.corrections import (CorrectionStore, apply_corrections, build_edit_items, propose, select_records, split_box,
                                       validate_boxes)
from courtdet.data.dataset import load_coco_records
from test_dataset import write_coco  # pytest puts tests/ on sys.path

_spec = importlib.util.spec_from_file_location("edit_server", Path(__file__).resolve().parents[1] / "scripts" / "edit_server.py")
edit_server = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(edit_server)

VALID = [
    ("basketball_court_1_jpg.rf.aaa.png", 256, 256, [(1, [10, 20, 120, 60])]),   # COCO [x, y, w, h]: wide box
    ("basketball_court_2_jpg.rf.bbb.png", 256, 256, [(1, [0, 0, 50, 50])]),
    ("basketball_court_3_jpg.rf.ccc.png", 256, 256, [(1, [5, 5, 40, 100])]),     # tall box
    ("P0001_png.rf.ddd.png", 600, 700, [(2, [10, 10, 50, 50])]),                 # source B: never edited
]
TRAIN = [
    ("basketball_court_7_jpg.rf.zzz.png", 256, 256, [(1, [30, 30, 100, 40])]),
    ("basketball_court_7_jpg.rf.aaa.png", 256, 256, [(1, [0, 0, 100, 40])]),     # other copy, other boxes
]
DECISIONS = {
    "valid/basketball_court_1_jpg": {"decision": "multiple"},
    "valid/basketball_court_2_jpg": {"decision": "single"},
    "valid/basketball_court_3_jpg": {"decision": "unsure"},
    "train/basketball_court_7_jpg": {"decision": "multiple"},
}


@pytest.fixture
def root(tmp_path):
    write_coco(tmp_path, "valid", VALID)
    write_coco(tmp_path, "train", TRAIN)
    return tmp_path


def test_split_box_by_hand():
    assert split_box([0, 0, 90, 10], 3, "x") == [[0, 0, 30, 10], [30, 0, 60, 10], [60, 0, 90, 10]]
    assert split_box([10, 20, 30, 60], 2, "y") == [[10, 20, 30, 40], [10, 40, 30, 60]]
    assert split_box([1, 2, 3, 4], 1, "x") == [[1, 2, 3, 4]]
    with pytest.raises(ValueError):
        split_box([0, 0, 1, 1], 0, "x")
    with pytest.raises(ValueError):
        split_box([0, 0, 1, 1], 2, "z")


def test_split_parts_tile_the_box():
    # Parts have equal areas, do not overlap and cover the box exactly.
    rng = np.random.default_rng(0)
    for _ in range(200):
        x1, y1 = rng.uniform(0, 100, 2)
        box = [x1, y1, x1 + rng.uniform(1, 150), y1 + rng.uniform(1, 150)]
        for n in (2, 3, 4):
            for along in ("x", "y"):
                parts = np.array(split_box(box, n, along))
                areas = (parts[:, 2] - parts[:, 0]) * (parts[:, 3] - parts[:, 1])
                assert np.allclose(areas, areas[0])
                assert np.isclose(areas.sum(), (box[2] - box[0]) * (box[3] - box[1]))
                assert np.allclose(parts.min(0)[:2], box[:2]) and np.allclose(parts.max(0)[2:], box[2:])


def test_propose_long_and_short_side():
    wide, tall = [0, 0, 100, 20], [0, 0, 20, 100]
    assert propose([wide], 2, "long") == [[0, 0, 50, 20], [50, 0, 100, 20]]     # cut across the long side
    assert propose([wide], 2, "short") == [[0, 0, 100, 10], [0, 10, 100, 20]]
    assert propose([tall], 2, "long") == [[0, 0, 20, 50], [0, 50, 20, 100]]
    assert propose([wide, tall], 1, "long") == [wide, tall]                     # n = 1: original boxes
    assert len(propose([wide, tall], 3, "long")) == 6


def test_edit_items_keep_multiple_and_unsure_only(root):
    items = build_edit_items(root, DECISIONS, splits=("valid", "train"))
    assert [it["id"] for it in items] == ["valid/basketball_court_1_jpg", "valid/basketball_court_3_jpg",
                                          "train/basketball_court_7_jpg"]
    first = items[0]
    assert first["key"] == "valid/basketball_court_1_jpg.rf.aaa.png"
    assert first["boxes"] == [[10.0, 20.0, 130.0, 80.0]]
    assert first["proposals"]["2-long"] == [[10.0, 20.0, 70.0, 80.0], [70.0, 20.0, 130.0, 80.0]]
    assert first["proposals"]["1-long"] == first["boxes"]
    assert set(first["proposals"]) == {f"{n}-{s}" for n in (1, 2, 3, 4) for s in ("long", "short")}
    # Train: only the first copy by file name is edited (the copies' boxes differ).
    assert items[2]["file_name"] == "basketball_court_7_jpg.rf.aaa.png"


def test_validate_boxes():
    assert validate_boxes([[0, 0, 256, 256], [1.234, 2, 10, 12]], 256, 256) == [[0, 0, 256, 256], [1.23, 2, 10, 12]]
    for bad in ([[0, 0, 257, 10]], [[-1, 0, 10, 10]], [[10, 10, 11, 20]], [[0, 0, 10]], [[20, 0, 10, 10]]):
        with pytest.raises(ValueError):
            validate_boxes(bad, 256, 256)


def test_store_and_apply(root, tmp_path):
    path = tmp_path / "review" / "corrections.json"
    store = CorrectionStore(path)
    store.set("valid/basketball_court_1_jpg.rf.aaa.png", "fixed", [[10, 20, 70, 80], [70, 20, 130, 80]], "split-2-long")
    store.set("valid/basketball_court_3_jpg.rf.ccc.png", "excluded", [[1, 1, 5, 5]])  # boxes of an excluded image dropped
    assert CorrectionStore(path).data["valid/basketball_court_3_jpg.rf.ccc.png"]["boxes"] == []
    assert store.counts() == {"fixed": 1, "excluded": 1}
    with pytest.raises(ValueError):
        store.set("valid/x", "fixed", [])
    with pytest.raises(ValueError):
        store.set("valid/x", "maybe", [[0, 0, 5, 5]])

    records = load_coco_records(root / "valid" / "_annotations.coco.json")
    out = apply_corrections(records, CorrectionStore(path).data, "valid")
    names = [r["file_name"] for r in out]
    assert "basketball_court_3_jpg.rf.ccc.png" not in names and len(out) == len(records) - 1
    fixed = out[names.index("basketball_court_1_jpg.rf.aaa.png")]
    np.testing.assert_array_equal(fixed["boxes"], [[10, 20, 70, 80], [70, 20, 130, 80]])
    untouched = out[names.index("basketball_court_2_jpg.rf.bbb.png")]
    np.testing.assert_array_equal(untouched["boxes"], [[0, 0, 50, 50]])
    assert records[0]["boxes"].shape == (1, 4)  # input records unchanged
    assert apply_corrections(records, CorrectionStore(path).data, "test") == records  # keys carry the split

    store.set("valid/basketball_court_1_jpg.rf.aaa.png", None)  # undo
    assert "valid/basketball_court_1_jpg.rf.aaa.png" not in CorrectionStore(path).data


def test_edit_server_end_to_end(root, tmp_path):
    decisions = tmp_path / "decisions.json"
    decisions.write_text(json.dumps(DECISIONS))
    server = edit_server.make_server(root, decisions, tmp_path / "c.json", port=0, splits=("valid",))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        assert b"Box editor" in urllib.request.urlopen(base + "/").read()
        items = json.loads(urllib.request.urlopen(base + "/api/items").read())
        assert [it["id"] for it in items] == ["valid/basketball_court_1_jpg", "valid/basketball_court_3_jpg"]
        assert urllib.request.urlopen(f"{base}/img/valid/{items[0]['file_name']}").read()[:4] == b"\x89PNG"
        for bad in ("/img/valid/basketball_court_2_jpg.rf.bbb.png",  # "single": not in the edit list
                    "/img/valid/../valid/_annotations.coco.json"):
            with pytest.raises(urllib.error.HTTPError) as err:
                urllib.request.urlopen(base + bad)
            assert err.value.code == 404

        def post(body):
            req = urllib.request.Request(base + "/api/correction", data=json.dumps(body).encode(), method="POST",
                                         headers={"Content-Type": "application/json"})
            return json.loads(urllib.request.urlopen(req).read())

        out = post({"id": items[0]["id"], "status": "fixed", "boxes": items[0]["proposals"]["2-long"],
                    "method": "split-2-long"})
        assert out[items[0]["key"]]["boxes"] == [[10.0, 20.0, 70.0, 80.0], [70.0, 20.0, 130.0, 80.0]]
        assert json.loads((tmp_path / "c.json").read_text())[items[0]["key"]]["method"] == "split-2-long"
        for bad in ({"id": "valid/unknown", "status": "fixed", "boxes": [[0, 0, 5, 5]]},
                    {"id": items[0]["id"], "status": "fixed", "boxes": [[0, 0, 300, 5]]},   # leaves the image
                    {"id": items[0]["id"], "status": "fixed", "boxes": []},
                    {"id": items[0]["id"], "status": "maybe", "boxes": [[0, 0, 5, 5]]}):
            with pytest.raises(urllib.error.HTTPError) as err:
                post(bad)
            assert err.value.code == 400
        assert post({"id": items[1]["id"], "status": "excluded"})[items[1]["key"]]["status"] == "excluded"
        assert items[1]["key"] not in post({"id": items[1]["id"], "status": None})
    finally:
        server.shutdown()
        server.server_close()


# --- record selection (corrected ground truth, dropped images, source A fraction) ---

SEL_TRAIN = [
    ("basketball_court_1_jpg.rf.a1.png", 256, 256, [(1, [0, 0, 50, 50])]),     # single
    ("basketball_court_1_jpg.rf.b1.png", 256, 256, [(1, [5, 0, 50, 50])]),
    ("basketball_court_2_jpg.rf.a2.png", 256, 256, [(1, [0, 0, 200, 100])]),   # multiple, this copy corrected
    ("basketball_court_2_jpg.rf.b2.png", 256, 256, [(1, [10, 0, 200, 100])]),  # multiple, other copy: dropped
    ("basketball_court_3_jpg.rf.a3.png", 256, 256, [(1, [0, 0, 200, 100])]),   # multiple, uncorrected: dropped
    ("basketball_court_3_jpg.rf.b3.png", 256, 256, [(1, [0, 0, 200, 100])]),
    ("basketball_court_4_jpg.rf.a4.png", 256, 256, [(1, [0, 0, 100, 100])]),   # unsure, uncorrected: dropped
    ("basketball_court_4_jpg.rf.b4.png", 256, 256, [(1, [0, 0, 100, 100])]),
    ("basketball_court_5_jpg.rf.a5.png", 256, 256, [(1, [0, 0, 60, 60])]),     # no decision at all: kept
    ("basketball_court_5_jpg.rf.b5.png", 256, 256, [(1, [0, 0, 60, 60])]),
    ("P0001_png.rf.p1.png", 600, 700, [(2, [10, 10, 50, 50])]),                # source B: kept
    ("P0002_png.rf.p2.png", 600, 700, []),
]
SEL_DECISIONS = {"train/basketball_court_1_jpg": {"decision": "single"},
                 "train/basketball_court_2_jpg": {"decision": "multiple"},
                 "train/basketball_court_3_jpg": {"decision": "multiple"},
                 "train/basketball_court_4_jpg": {"decision": "unsure"}}
SEL_CORRECTIONS = {"train/basketball_court_2_jpg.rf.a2.png": {"status": "fixed", "boxes": [[0, 0, 100, 100], [100, 0, 200, 100]]}}


@pytest.fixture
def sel_root(tmp_path):
    write_coco(tmp_path, "train", SEL_TRAIN)
    write_coco(tmp_path, "valid", VALID)
    (tmp_path / "decisions.json").write_text(json.dumps(SEL_DECISIONS))
    (tmp_path / "corrections.json").write_text(json.dumps(SEL_CORRECTIONS))
    return tmp_path


def names(records):
    return sorted(r["file_name"].split(".rf.")[1] for r in records)


def test_select_records_steps(sel_root):
    records = load_coco_records(sel_root / "train" / "_annotations.coco.json")
    assert select_records(records, "train") == records  # nothing given: unchanged
    only_corr = select_records(records, "train", corrections=SEL_CORRECTIONS)
    assert len(only_corr) == len(records)
    out = select_records(records, "train", corrections=SEL_CORRECTIONS, decisions=SEL_DECISIONS)
    assert names(out) == ["a1.png", "a2.png", "a5.png", "b1.png", "b5.png", "p1.png", "p2.png"]
    a2 = next(r for r in out if r["file_name"].endswith("a2.png"))
    np.testing.assert_array_equal(a2["boxes"], [[0, 0, 100, 100], [100, 0, 200, 100]])
    # decisions without corrections: every multiple/unsure A image goes, corrected or not
    assert names(select_records(records, "train", decisions=SEL_DECISIONS)) == [
        "a1.png", "a5.png", "b1.png", "b5.png", "p1.png", "p2.png"]


def test_source_a_fraction_keeps_whole_sources(sel_root):
    records = load_coco_records(sel_root / "train" / "_annotations.coco.json")  # 5 A sources, 2 copies each
    seen = set()
    for seed in range(20):
        out = select_records(records, "train", source_a_fraction=0.4, seed=seed)
        a = [r for r in out if r["source"] == "A"]
        keys = {r["file_name"].split(".rf.")[0] for r in a}
        assert len(keys) == 2 and len(a) == 4  # round(0.4 * 5) sources, both copies of each
        assert sum(r["source"] == "B" for r in out) == 2  # B untouched
        assert out == select_records(records, "train", source_a_fraction=0.4, seed=seed)  # reproducible
        seen.add(frozenset(keys))
    assert len(seen) > 1  # the seed matters
    with pytest.raises(ValueError):
        select_records(records, "train", source_a_fraction=0.0)


def test_load_records_and_loaders_use_the_config(sel_root):
    from courtdet.config import Config
    from courtdet.engine.train import make_loaders
    from courtdet.data.corrections import load_records

    recs = load_records(sel_root, "train", corrections=sel_root / "corrections.json",
                        decisions=sel_root / "decisions.json")
    assert names(recs) == ["a1.png", "a2.png", "a5.png", "b1.png", "b5.png", "p1.png", "p2.png"]

    config = Config()
    config.data.root = str(sel_root)
    config.data.num_workers = 0
    train, val = make_loaders(config)
    assert len(train.dataset) == len(SEL_TRAIN) and len(val.dataset) == len(VALID)  # defaults: no selection
    config.data.corrections = str(sel_root / "corrections.json")
    config.data.decisions = str(sel_root / "decisions.json")
    config.data.source_a_fraction = 0.5
    train, _ = make_loaders(config)
    a_keys = {r["file_name"].split(".rf.")[0] for r in train.dataset.records if r["source"] == "A"}
    assert len(a_keys) == 2 and sum(r["source"] == "B" for r in train.dataset.records) == 2  # round(0.5 * 3)
