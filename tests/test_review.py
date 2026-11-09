import importlib.util
import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from courtdet.data.review import DecisionStore, build_items, source_key
from test_dataset import write_coco  # pytest puts tests/ on sys.path

_spec = importlib.util.spec_from_file_location("review_server", Path(__file__).resolve().parents[1] / "scripts" / "review_server.py")
review_server = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(review_server)

TRAIN = [
    ("basketball_court_1_jpg.rf.aaa.png", 256, 256, [(1, [10, 10, 100, 60])]),
    ("basketball_court_1_jpg.rf.bbb.png", 256, 256, [(1, [20, 10, 110, 60])]),  # 2nd copy of the same source
    ("basketball_court_2_jpg.rf.ccc.png", 256, 256, [(1, [0, 0, 50, 50]), (1, [100, 100, 150, 150])]),
    ("P0001_png.rf.ddd.png", 600, 700, [(2, [10, 10, 50, 50])]),  # source B: not reviewed
]
VALID = [("basketball_court_9_jpg.rf.eee.png", 256, 256, [(1, [5, 5, 60, 60])])]


@pytest.fixture
def root(tmp_path):
    write_coco(tmp_path, "train", TRAIN)
    write_coco(tmp_path, "valid", VALID)
    return tmp_path


def test_source_key():
    assert source_key("basketball_court_523_jpg.rf.eea455.jpg") == "basketball_court_523_jpg"


def test_items_group_copies_and_skip_source_b(root):
    items = build_items(root, splits=("valid", "train"))
    assert [it["id"] for it in items] == ["valid/basketball_court_9_jpg", "train/basketball_court_1_jpg",
                                          "train/basketball_court_2_jpg"]
    assert len(items[1]["images"]) == 2  # both copies, one decision
    assert items[2]["images"][0]["boxes"] == [[0.0, 0.0, 50.0, 50.0], [100.0, 100.0, 250.0, 250.0]]  # COCO [x, y, w, h] -> corners


def test_decision_store_persists_and_validates(tmp_path):
    path = tmp_path / "review" / "decisions.json"
    store = DecisionStore(path)
    store.set("train/x", "multiple")
    store.set("valid/y", "single")
    assert DecisionStore(path).data.keys() == {"train/x", "valid/y"}  # reloaded from disk
    assert store.counts() == {"single": 1, "multiple": 1, "unsure": 0}
    store.set("train/x", None)
    assert DecisionStore(path).data.keys() == {"valid/y"}
    with pytest.raises(ValueError):
        store.set("valid/y", "maybe")
    assert not list(path.parent.glob(".decisions-*"))  # no temporary file left behind


def test_server_end_to_end(root, tmp_path):
    server = review_server.make_server(root, tmp_path / "d.json", port=0, splits=("valid", "train"))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        assert b"Court review" in urllib.request.urlopen(base + "/").read()
        items = json.loads(urllib.request.urlopen(base + "/api/items").read())
        assert len(items) == 3

        name = items[1]["images"][0]["file_name"]
        img = urllib.request.urlopen(f"{base}/img/train/{name}").read()
        assert img[:4] == b"\x89PNG"
        for bad in ("/img/train/P0001_png.rf.ddd.png", "/img/train/../train/_annotations.coco.json", "/img/x/y"):
            with pytest.raises(urllib.error.HTTPError) as err:
                urllib.request.urlopen(base + bad)
            assert err.value.code == 404

        def post(body):
            req = urllib.request.Request(base + "/api/decision", data=json.dumps(body).encode(), method="POST",
                                         headers={"Content-Type": "application/json"})
            return json.loads(urllib.request.urlopen(req).read())

        assert post({"id": items[0]["id"], "decision": "multiple"})[items[0]["id"]]["decision"] == "multiple"
        assert json.loads((tmp_path / "d.json").read_text())[items[0]["id"]]["decision"] == "multiple"
        for bad in ({"id": "train/unknown", "decision": "single"}, {"id": items[0]["id"], "decision": "maybe"}):
            with pytest.raises(urllib.error.HTTPError) as err:
                post(bad)
            assert err.value.code == 400
    finally:
        server.shutdown()
        server.server_close()


def test_concurrent_decisions_are_all_saved(tmp_path):
    # The server is multi-threaded: many simultaneous saves must neither crash nor lose a decision.
    store = DecisionStore(tmp_path / "d.json")
    errors = []

    def worker(k):
        try:
            for j in range(20):
                store.set(f"train/item_{k}_{j}", "single" if j % 2 else "multiple")
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(8)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert errors == []
    assert len(DecisionStore(tmp_path / "d.json").data) == 8 * 20
