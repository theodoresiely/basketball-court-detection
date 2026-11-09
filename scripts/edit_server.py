"""Local box editor, step 2: one box per court for the source A images sorted "multiple" / "unsure".

    python scripts/edit_server.py            # then open http://127.0.0.1:8766

Each image starts from a geometric proposal: every original box cut into 2 equal parts along its
long side (courtdet.data.corrections.propose). Keys:
  1-4      original box cut into n parts (1 = original box)      r      cut along the other side
  drag an edge / corner   resize that box      Shift + drag inside a box   move it
  drag elsewhere          draw a new box       click a box + Delete        remove it      c   remove every box
  Enter    save as "fixed" (the boxes shown are right)            x      exclude (half court / unusable)
  u        remove the saved correction    <- / ->  previous / next    n  next image not saved
Corrections go to data/review/corrections.json; the Roboflow annotations are not modified.
Listens on 127.0.0.1 only; serves only the images of the edit list.
"""

from __future__ import annotations

import argparse
import json
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

from courtdet.data.corrections import CorrectionStore, build_edit_items, validate_boxes

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Box editor</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  body { margin: 0; background: #16181c; color: #e6e6e6; font: 14px system-ui, sans-serif; }
  header { padding: 10px 16px; display: flex; gap: 18px; align-items: center; flex-wrap: wrap; border-bottom: 1px solid #333; }
  #bar { flex: 1; min-width: 160px; height: 8px; background: #333; border-radius: 4px; overflow: hidden; }
  #fill { height: 100%; background: #3fb950; width: 0; }
  main { display: flex; gap: 24px; padding: 16px; justify-content: center; align-items: flex-start; flex-wrap: wrap; }
  .frame { position: relative; display: inline-block; user-select: none; }
  /* 256 px tiles upscaled: boxes are drawn on a readable image */
  .frame img { display: block; width: min(90vw, 78vh, 760px); height: auto; image-rendering: auto; }
  .frame svg { position: absolute; inset: 0; width: 100%; height: 100%; cursor: crosshair; }
  aside { max-width: 300px; line-height: 1.6; }
  #state { font-weight: 600; }
  .fixed { color: #3fb950; } .excluded { color: #f85149; } .todo { color: #d29922; }
  kbd { background: #333; border-radius: 3px; padding: 1px 5px; }
  #boxes { font-family: ui-monospace, monospace; font-size: 12px; color: #aaa; white-space: pre; }
</style></head><body>
<header>
  <span id="pos"></span><span id="id"></span><span id="state"></span>
  <div id="bar"><div id="fill"></div></div><span id="counts"></span>
</header>
<main>
  <div class="frame"><img id="img" draggable="false"><svg id="svg"></svg></div>
  <aside>
    <div>step 1: <b id="decision"></b> &middot; boxes: <b id="method"></b></div>
    <div id="boxes"></div>
    <hr>
    <div><kbd>1</kbd>-<kbd>4</kbd> original box cut in n parts (<kbd>1</kbd> = original)</div>
    <div><kbd>r</kbd> cut along the other side</div>
    <div>drag an edge or a corner = resize the box</div>
    <div><kbd>Shift</kbd> + drag inside a box = move it</div>
    <div>drag elsewhere (even inside a box) = draw a new box</div>
    <div>click a box + <kbd>&#9003;</kbd> (delete) = remove it</div>
    <div><kbd>c</kbd> remove every box</div>
    <div><kbd>Enter</kbd> save: boxes are right</div>
    <div><kbd>x</kbd> exclude (half court / unusable)</div>
    <div><kbd>u</kbd> remove the saved correction</div>
    <div><kbd>&larr;</kbd> <kbd>&rarr;</kbd> previous / next &middot; <kbd>n</kbd> next not saved</div>
    <div style="color:#888;margin-top:8px">dashed grey = original box &middot; green = boxes to save &middot; yellow = selected</div>
  </aside>
</main>
<script>
let items = [], saved = {}, i = 0;
let boxes = [], method = "", n = 2, side = "long", selected = -1, drawing = null;
const $ = (s) => document.querySelector(s);
const NS = "http://www.w3.org/2000/svg";

async function load() {
  items = await (await fetch("/api/items")).json();
  saved = await (await fetch("/api/corrections")).json();
  i = Math.max(0, items.findIndex((it) => !saved[it.key]));
  open();
}

// Working boxes of the current image: the saved correction if any, else the 2-part proposal.
function open() {
  const it = items[i], s = saved[it.key];
  n = 2; side = "long"; selected = -1;
  if (s && s.status === "fixed") { boxes = s.boxes.map((b) => b.slice()); method = s.method; }
  else if (s) { boxes = []; method = s.method; }
  else { applyProposal(); return; }
  show();
}

function applyProposal() {
  boxes = items[i].proposals[`${n}-${side}`].map((b) => b.slice());
  method = n === 1 ? "original" : `split-${n}-${side}`;
  selected = -1;
  show();
}

function rect(svg, [x1, y1, x2, y2], attrs) {
  const r = document.createElementNS(NS, "rect");
  Object.entries({ x: x1, y: y1, width: x2 - x1, height: y2 - y1, fill: "none", ...attrs })
    .forEach(([a, v]) => r.setAttribute(a, v));
  svg.appendChild(r);
  return r;
}

function show() {
  const it = items[i], s = saved[it.key];
  $("#pos").textContent = `${i + 1} / ${items.length}`;
  $("#id").textContent = it.id;
  const st = $("#state");
  st.textContent = s ? `saved: ${s.status}` : "not saved";
  st.className = s ? s.status : "todo";
  $("#decision").textContent = it.decision;
  $("#method").textContent = method;
  $("#boxes").textContent = boxes.map((b) => b.map((v) => v.toFixed(1)).join(", ")).join("\n") || "(no box)";

  const done = items.filter((x) => saved[x.key]).length;
  $("#fill").style.width = `${(100 * done) / items.length}%`;
  const c = { fixed: 0, excluded: 0, kept: 0, adjusted: 0, manual: 0 };
  Object.values(saved).forEach((v) => {
    c[v.status]++;
    if (v.status === "fixed") c[v.method === "manual" ? "manual" : v.method.endsWith("+adjusted") ? "adjusted" : "kept"]++;
  });
  $("#counts").textContent = `${done} saved - fixed ${c.fixed} (proposal kept ${c.kept}, adjusted ${c.adjusted}, redrawn ${c.manual}) - excluded ${c.excluded}`;

  const img = $("#img");
  const src = `/img/${it.split}/${encodeURIComponent(it.file_name)}`;
  if (img.getAttribute("src") !== src) img.setAttribute("src", src);
  const svg = $("#svg");
  svg.innerHTML = "";
  svg.setAttribute("viewBox", `0 0 ${it.width} ${it.height}`);
  svg.setAttribute("preserveAspectRatio", "none");
  const sw = it.width / 300;
  it.boxes.forEach((b) => rect(svg, b, { stroke: "#9aa0a6", "stroke-width": sw, "stroke-dasharray": `${4 * sw} ${3 * sw}` }));
  boxes.forEach((b, k) => {
    rect(svg, b, { stroke: k === selected ? "#f0c000" : "#28dc3c", "stroke-width": 1.6 * sw, "pointer-events": "none" });
    if (k === selected) {  // corner handles of the selected box
      const h = 6 * sw;
      [[b[0], b[1]], [b[2], b[1]], [b[0], b[3]], [b[2], b[3]]].forEach(([x, y]) =>
        rect(svg, [x - h / 2, y - h / 2, x + h / 2, y + h / 2], { fill: "#f0c000", "pointer-events": "none" }));
    }
  });
  if (drawing) rect(svg, norm(drawing), { stroke: "#58a6ff", "stroke-width": 1.6 * sw, "pointer-events": "none" });
}

// Mouse position in image pixels, clamped to the image.
function toImage(e) {
  const svg = $("#svg"), it = items[i];
  const p = svg.createSVGPoint();
  p.x = e.clientX; p.y = e.clientY;
  const q = p.matrixTransform(svg.getScreenCTM().inverse());
  return [Math.min(Math.max(q.x, 0), it.width), Math.min(Math.max(q.y, 0), it.height)];
}
const norm = ([ax, ay, bx, by]) => [Math.min(ax, bx), Math.min(ay, by), Math.max(ax, bx), Math.max(ay, by)];

// What a press at (x, y) grabs: the edges of a box within `tol` image pixels (a corner grabs 2 edges),
// the selected box first. edges = indices into [x1, y1, x2, y2].
function grab(x, y) {
  const tol = 8 * items[i].width / $("#svg").getBoundingClientRect().width;  // 8 screen pixels
  const order = boxes.map((_, k) => k).sort((a, b) => (b === selected) - (a === selected));
  for (const k of order) {
    const [x1, y1, x2, y2] = boxes[k];
    if (x < x1 - tol || x > x2 + tol || y < y1 - tol || y > y2 + tol) continue;
    const edges = [];
    if (Math.abs(x - x1) <= tol) edges.push(0); else if (Math.abs(x - x2) <= tol) edges.push(2);
    if (Math.abs(y - y1) <= tol) edges.push(1); else if (Math.abs(y - y2) <= tol) edges.push(3);
    if (edges.length) return { k, edges };
  }
  return null;
}
// Smallest box containing (x, y), or -1.
function boxAt(x, y) {
  let best = -1, area = Infinity;
  boxes.forEach(([x1, y1, x2, y2], k) => {
    const a = (x2 - x1) * (y2 - y1);
    if (x >= x1 && x <= x2 && y >= y1 && y <= y2 && a < area) { best = k; area = a; }
  });
  return best;
}
const CURSORS = { "0": "ew-resize", "2": "ew-resize", "1": "ns-resize", "3": "ns-resize",
                  "0,1": "nwse-resize", "2,3": "nwse-resize", "2,1": "nesw-resize", "0,3": "nesw-resize" };
const round1 = (b) => b.map((v) => Math.round(v * 10) / 10);
function adjusted() { if (method !== "manual" && !method.endsWith("+adjusted")) method += "+adjusted"; }

let drag = null;  // {kind: "draw" | "resize" | "move", ...}
$("#svg").addEventListener("pointerdown", (e) => {
  const [x, y] = toImage(e), g = grab(x, y), inside = boxAt(x, y);
  if (g) { drag = { kind: "resize", ...g }; selected = g.k; }
  else if (e.shiftKey && inside >= 0) { drag = { kind: "move", k: inside, from: [x, y], start: boxes[inside].slice() }; selected = inside; }
  else { drag = { kind: "draw", from: [x, y] }; drawing = [x, y, x, y]; }
  $("#svg").setPointerCapture(e.pointerId);
  show();
});
$("#svg").addEventListener("pointermove", (e) => {
  const [x, y] = toImage(e), svg = $("#svg");
  if (!drag) {  // hover: show what a press would do
    const g = grab(x, y);
    svg.style.cursor = g ? CURSORS[g.edges.join(",")] : e.shiftKey && boxAt(x, y) >= 0 ? "move" : "crosshair";
    return;
  }
  if (drag.kind === "draw") { drawing[2] = x; drawing[3] = y; }
  else if (drag.kind === "resize") { drag.edges.forEach((j) => { boxes[drag.k][j] = j % 2 ? y : x; }); drag.moved = true; }
  else {
    const it = items[i], [x1, y1, x2, y2] = drag.start;
    const dx = Math.min(Math.max(x - drag.from[0], -x1), it.width - x2);  // keep the box inside the image
    const dy = Math.min(Math.max(y - drag.from[1], -y1), it.height - y2);
    boxes[drag.k] = [x1 + dx, y1 + dy, x2 + dx, y2 + dy]; drag.moved = true;
  }
  show();
});
$("#svg").addEventListener("pointerup", (e) => {
  if (!drag) return;
  const d = drag;
  drag = null;
  if (d.kind === "draw") {
    const b = round1(norm(drawing));
    drawing = null;
    if (b[2] - b[0] >= 2 && b[3] - b[1] >= 2) { boxes.push(b); selected = boxes.length - 1; method = "manual"; }
    else { const [x, y] = toImage(e); selected = boxAt(x, y); }  // a click, not a box: select (or deselect)
  } else if (d.moved) {
    const b = round1(norm(boxes[d.k]));  // an edge dragged past the opposite one flips back to a valid box
    if (b[2] - b[0] >= 2 && b[3] - b[1] >= 2) boxes[d.k] = b; else { boxes.splice(d.k, 1); selected = -1; }
    adjusted();
  }
  show();
});

// The correction is bound to the image shown WHEN THE KEY IS PRESSED; saves are sent in order.
let saving = Promise.resolve();
function save(status) {
  const it = items[i];
  if (status === "fixed" && !boxes.length) { alert("no box: draw one, or exclude the image with x"); return; }
  const body = { id: it.id, status, boxes: status === "fixed" ? boxes : [], method: status === "fixed" ? method : "" };
  if (status) saved[it.key] = { status, boxes: body.boxes.map((b) => b.slice()), method: body.method };
  else delete saved[it.key];
  if (status && i < items.length - 1) { i++; open(); } else open();
  saving = saving.then(async () => {
    const res = await fetch("/api/correction", { method: "POST", headers: { "Content-Type": "application/json" },
                                                  body: JSON.stringify(body) });
    if (!res.ok) {
      alert(`save failed for ${it.id}: ` + (await res.text()));
      saved = await (await fetch("/api/corrections")).json();  // back to what is really on disk
      show();
    }
  });
}

function go(k) { if (k >= 0 && k < items.length) { i = k; open(); } }

document.addEventListener("keydown", (e) => {
  if (["1", "2", "3", "4"].includes(e.key)) { n = Number(e.key); applyProposal(); }
  else if (e.key === "r") { side = side === "long" ? "short" : "long"; applyProposal(); }
  else if ((e.key === "Delete" || e.key === "Backspace") && selected >= 0) {
    e.preventDefault(); boxes.splice(selected, 1); selected = -1; method = "manual"; show();
  }
  else if (e.key === "c") { boxes = []; selected = -1; method = "manual"; show(); }
  else if (e.key === "Enter") save("fixed");
  else if (e.key === "x") save("excluded");
  else if (e.key === "u") save(null);
  else if (e.key === "ArrowLeft") go(i - 1);
  else if (e.key === "ArrowRight") go(i + 1);
  else if (e.key === "n") go(items.findIndex((it, j) => j > i && !saved[it.key]));
});
load();
</script></body></html>
"""


def make_handler(root: Path, items: list[dict], store: CorrectionStore):
    allowed = {(it["split"], it["file_name"]) for it in items}
    by_id = {it["id"]: it for it in items}

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, status: int = HTTPStatus.OK) -> None:
            self._send(status, json.dumps(obj).encode(), "application/json")

        def do_GET(self) -> None:  # noqa: N802 (http.server API)
            path = unquote(self.path.split("?")[0])
            if path == "/":
                self._send(HTTPStatus.OK, PAGE.encode(), "text/html; charset=utf-8")
            elif path == "/api/items":
                self._json(items)
            elif path == "/api/corrections":
                self._json(store.snapshot())
            elif path.startswith("/img/"):
                split, _, file_name = path[len("/img/"):].partition("/")
                if (split, file_name) not in allowed:  # only listed images: no path traversal
                    self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")
                    return
                data = (root / split / file_name).read_bytes()
                kind = "image/png" if file_name.lower().endswith(".png") else "image/jpeg"
                self._send(HTTPStatus.OK, data, kind)
            else:
                self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/api/correction":
                self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")
                return
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                it = by_id.get(body.get("id"))
                if it is None:
                    raise ValueError(f"unknown item {body.get('id')!r}")
                status = body.get("status")
                boxes = validate_boxes(body.get("boxes") or [], it["width"], it["height"]) if status == "fixed" else []
                store.set(it["key"], status, boxes, str(body.get("method", "")))
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                self._send(HTTPStatus.BAD_REQUEST, str(exc).encode(), "text/plain")
                return
            self._json(store.snapshot())

        def log_message(self, fmt, *args) -> None:  # quieter: only errors
            if args and str(args[1]).startswith(("4", "5")):
                super().log_message(fmt, *args)

    return Handler


def make_server(root: Path, decisions: Path, corrections: Path, host: str = "127.0.0.1", port: int = 8766,
                splits=("valid", "test")) -> ThreadingHTTPServer:
    items = build_edit_items(root, json.loads(Path(decisions).read_text()), splits)
    store = CorrectionStore(corrections)
    return ThreadingHTTPServer((host, port), make_handler(root, items, store))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="dataset/basket_dataset.coco", type=Path)
    parser.add_argument("--decisions", default="data/review/decisions.json", type=Path)
    parser.add_argument("--corrections", default="data/review/corrections.json", type=Path)
    parser.add_argument("--splits", nargs="+", default=["valid", "test"])
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args(argv)
    server = make_server(args.root, args.decisions, args.corrections, port=args.port, splits=tuple(args.splits))
    print(f"box editor: http://127.0.0.1:{server.server_address[1]}  (corrections -> {args.corrections})", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main(sys.argv[1:])
