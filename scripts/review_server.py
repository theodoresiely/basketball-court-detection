"""Local review page, step 1: sort every original source A image into single / multiple / unsure.

    python scripts/review_server.py            # then open http://127.0.0.1:8765

Keys: 1 = one court per box, 2 = a box holds several courts, 3 = unsure, 0 = clear the decision,
      <- / -> = previous / next, n = next image without a decision.
Decisions are saved on every key press to data/review/decisions.json (see courtdet.data.review).
Listens on 127.0.0.1 only; serves only the images of the review list.
"""

from __future__ import annotations

import argparse
import json
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

from courtdet.data.review import DecisionStore, build_items

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Court review</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  body { margin: 0; background: #16181c; color: #e6e6e6; font: 14px system-ui, sans-serif; }
  header { padding: 10px 16px; display: flex; gap: 18px; align-items: center; flex-wrap: wrap; border-bottom: 1px solid #333; }
  #bar { flex: 1; min-width: 160px; height: 8px; background: #333; border-radius: 4px; overflow: hidden; }
  #fill { height: 100%; background: #3fb950; width: 0; }
  main { display: flex; gap: 16px; padding: 16px; justify-content: center; flex-wrap: wrap; }
  figure { margin: 0; text-align: center; }
  figcaption { color: #999; font-size: 12px; margin-top: 4px; }
  .frame { position: relative; display: inline-block; }
  /* 256 px tiles are upscaled: small courts must be judged at a readable size */
  .frame img { display: block; width: min(44vw, 72vh, 600px); height: auto; }
  .frame svg { position: absolute; inset: 0; width: 100%; height: 100%; }
  footer { display: flex; gap: 10px; justify-content: center; padding: 8px 16px 20px; flex-wrap: wrap; }
  button { font: inherit; padding: 8px 14px; border-radius: 6px; border: 1px solid #555; background: #24272d; color: #e6e6e6; cursor: pointer; }
  button.on { outline: 3px solid #f0c000; }
  .single { border-color: #3fb950; } .multiple { border-color: #f85149; } .unsure { border-color: #d29922; }
  #state { font-weight: 600; }
  kbd { background: #333; border-radius: 3px; padding: 1px 5px; }
</style></head><body>
<header>
  <span id="pos"></span><span id="id"></span><span id="state"></span>
  <div id="bar"><div id="fill"></div></div><span id="counts"></span>
</header>
<main id="view"></main>
<footer>
  <button class="single" data-d="single"><kbd>1</kbd> one court per box</button>
  <button class="multiple" data-d="multiple"><kbd>2</kbd> a box holds several courts</button>
  <button class="unsure" data-d="unsure"><kbd>3</kbd> unsure</button>
  <button data-d=""><kbd>0</kbd> clear</button>
  <button id="prev"><kbd>&larr;</kbd></button><button id="next"><kbd>&rarr;</kbd></button>
  <button id="todo"><kbd>n</kbd> next undecided</button>
</footer>
<script>
let items = [], decisions = {}, i = 0;
const $ = (s) => document.querySelector(s);

async function load() {
  items = await (await fetch("/api/items")).json();
  decisions = await (await fetch("/api/decisions")).json();
  i = Math.max(0, items.findIndex((it) => !decisions[it.id]));
  show();
}

function show() {
  const it = items[i];
  $("#pos").textContent = `${i + 1} / ${items.length}`;
  $("#id").textContent = it.id;
  const d = decisions[it.id]?.decision;
  $("#state").textContent = d ? `-> ${d}` : "(no decision)";
  document.querySelectorAll("button[data-d]").forEach((b) => b.classList.toggle("on", b.dataset.d === (d || "-")));
  const done = items.filter((x) => decisions[x.id]).length;
  $("#fill").style.width = `${(100 * done) / items.length}%`;
  const c = { single: 0, multiple: 0, unsure: 0 };
  Object.values(decisions).forEach((v) => c[v.decision]++);
  $("#counts").textContent = `${done} done - single ${c.single} - multiple ${c.multiple} - unsure ${c.unsure}`;

  const view = $("#view");
  view.innerHTML = "";
  it.images.forEach((im) => {
    const fig = document.createElement("figure");
    const frame = document.createElement("div");
    frame.className = "frame";
    const img = document.createElement("img");
    img.src = `/img/${it.split}/${encodeURIComponent(im.file_name)}`;
    frame.appendChild(img);
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("viewBox", `0 0 ${im.width} ${im.height}`);
    svg.setAttribute("preserveAspectRatio", "none");
    im.boxes.forEach(([x1, y1, x2, y2], k) => {
      const r = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      Object.entries({ x: x1, y: y1, width: x2 - x1, height: y2 - y1, fill: "none",
        stroke: "#28dc3c", "stroke-width": Math.max(1.5, im.width / 200) }).forEach(([a, v]) => r.setAttribute(a, v));
      svg.appendChild(r);
    });
    frame.appendChild(svg);
    fig.appendChild(frame);
    const cap = document.createElement("figcaption");
    cap.textContent = `${im.file_name} - ${im.width}x${im.height} - ${im.boxes.length} box(es)`;
    fig.appendChild(cap);
    view.appendChild(fig);
  });
}

// The decision is bound to the image shown WHEN THE KEY IS PRESSED, the view moves on at once,
// and saves are sent one after the other. (Waiting for the server before moving on let fast
// key presses land on the same image.)
let saving = Promise.resolve();
function decide(d) {
  const id = items[i].id;
  if (d) decisions[id] = { decision: d }; else delete decisions[id];
  if (d && i < items.length - 1) i++;
  show();
  saving = saving.then(async () => {
    const res = await fetch("/api/decision", { method: "POST", headers: { "Content-Type": "application/json" },
                                                body: JSON.stringify({ id, decision: d || null }) });
    if (!res.ok) {
      alert(`save failed for ${id}: ` + (await res.text()));
      decisions = await (await fetch("/api/decisions")).json();  // back to what is really on disk
      show();
    }
  });
}

function nextUndecided() {
  const k = items.findIndex((it, j) => j > i && !decisions[it.id]);
  if (k >= 0) { i = k; show(); }
}

document.querySelectorAll("button[data-d]").forEach((b) => b.onclick = () => decide(b.dataset.d));
$("#prev").onclick = () => { if (i > 0) { i--; show(); } };
$("#next").onclick = () => { if (i < items.length - 1) { i++; show(); } };
$("#todo").onclick = nextUndecided;
document.addEventListener("keydown", (e) => {
  const map = { "1": "single", "2": "multiple", "3": "unsure", "0": "" };
  if (e.key in map) decide(map[e.key]);
  else if (e.key === "ArrowLeft") $("#prev").click();
  else if (e.key === "ArrowRight") $("#next").click();
  else if (e.key === "n") nextUndecided();
});
load();
</script></body></html>
"""


def make_handler(root: Path, items: list[dict], store: DecisionStore):
    allowed = {(it["split"], im["file_name"]) for it in items for im in it["images"]}
    item_ids = {it["id"] for it in items}

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
            elif path == "/api/decisions":
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
            if self.path != "/api/decision":
                self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")
                return
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                if body.get("id") not in item_ids:
                    raise ValueError(f"unknown item {body.get('id')!r}")
                store.set(body["id"], body.get("decision"))
            except (ValueError, json.JSONDecodeError) as exc:
                self._send(HTTPStatus.BAD_REQUEST, str(exc).encode(), "text/plain")
                return
            self._json(store.snapshot())

        def log_message(self, fmt, *args) -> None:  # quieter: only errors
            if args and str(args[1]).startswith(("4", "5")):
                super().log_message(fmt, *args)

    return Handler


def make_server(root: Path, decisions: Path, host: str = "127.0.0.1", port: int = 8765,
                splits=("valid", "test", "train")) -> ThreadingHTTPServer:
    items = build_items(root, splits)
    store = DecisionStore(decisions)
    return ThreadingHTTPServer((host, port), make_handler(root, items, store))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="dataset/basket_dataset.coco", type=Path)
    parser.add_argument("--decisions", default="data/review/decisions.json", type=Path)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    server = make_server(args.root, args.decisions, port=args.port)
    print(f"review page: http://127.0.0.1:{server.server_address[1]}  (decisions -> {args.decisions})", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main(sys.argv[1:])
