#!/usr/bin/env python3
"""Throwaway viewer for training dataset samples.

Usage:
    uv run python scripts/view_training_data.py
    uv run python scripts/view_training_data.py --port 8080
    uv run python scripts/view_training_data.py --file training/dataset/v2/processed/train.jsonl
"""

import argparse
import html
import json
import mimetypes
import re
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

REPO_ROOT = Path(__file__).resolve().parent.parent


def find_jsonl_files() -> list[Path]:
    """Return JSONL files, v2+ first (have real images), v1 last."""
    all_files = list(REPO_ROOT.glob("training/dataset/**/processed/*.jsonl"))
    v2_plus = sorted(f for f in all_files if "/v1/" not in str(f))
    v1 = sorted(f for f in all_files if "/v1/" in str(f))
    return v2_plus + v1


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# ── HTML helpers ──────────────────────────────────────────────────────────────

CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Segoe UI', system-ui, sans-serif; background: #0f1117; color: #e2e8f0; min-height: 100vh; }

.topbar {
    display: flex; align-items: center; gap: 12px; padding: 10px 20px;
    background: #1a1d27; border-bottom: 1px solid #2d3148; flex-wrap: wrap;
}
.topbar h1 { font-size: 14px; font-weight: 600; color: #818cf8; white-space: nowrap; }
.topbar select, .topbar input {
    background: #252839; border: 1px solid #3d4268; color: #e2e8f0;
    padding: 5px 8px; border-radius: 6px; font-size: 13px;
}
.topbar select { cursor: pointer; }
.nav { display: flex; align-items: center; gap: 8px; margin-left: auto; }
.btn {
    background: #3730a3; color: #fff; border: none; padding: 6px 14px;
    border-radius: 6px; cursor: pointer; font-size: 13px; text-decoration: none;
    white-space: nowrap;
}
.btn:hover { background: #4338ca; }
.btn.disabled { background: #2d3148; color: #555; pointer-events: none; }
.counter { font-size: 13px; color: #6b7280; white-space: nowrap; }

.main { display: grid; grid-template-columns: auto 1fr; gap: 0; height: calc(100vh - 53px); overflow: hidden; }

.images-panel {
    display: flex; flex-direction: column; gap: 8px;
    padding: 12px; background: #13161f; border-right: 1px solid #1e2130;
    overflow-y: auto; min-width: 220px; max-width: 300px;
}
.img-wrap { text-align: center; }
.img-wrap span { display: block; font-size: 10px; color: #6b7280; margin-bottom: 4px; }
.img-wrap img { width: 100%; border-radius: 6px; border: 1px solid #2d3148; }
.img-label {
    font-size: 10px; margin-top: 3px; padding: 2px 6px; border-radius: 4px;
    display: inline-block;
}
.label-normal { background: #1e3a5f; color: #93c5fd; }
.label-fragile { background: #3b1515; color: #fca5a5; }
.label-heavy { background: #3b2a0a; color: #fcd34d; }
.label-damaged { background: #2d1f3d; color: #c4b5fd; }

.content-panel { display: flex; flex-direction: column; overflow: hidden; }

.tabs { display: flex; border-bottom: 1px solid #1e2130; }
.tab {
    padding: 8px 18px; font-size: 13px; cursor: pointer; color: #6b7280;
    border-bottom: 2px solid transparent; background: none; border-top: none;
    border-left: none; border-right: none; white-space: nowrap;
}
.tab.active { color: #818cf8; border-bottom-color: #818cf8; }

.tab-content { display: none; flex: 1; overflow-y: auto; padding: 16px; }
.tab-content.active { display: block; }

pre {
    white-space: pre-wrap; word-break: break-word; font-size: 12px;
    line-height: 1.6; font-family: 'JetBrains Mono', 'Fira Code', monospace;
}

.section-label {
    font-size: 10px; font-weight: 600; text-transform: uppercase;
    letter-spacing: .08em; color: #6b7280; margin-bottom: 6px; margin-top: 14px;
}
.section-label:first-child { margin-top: 0; }

.prompt-text { color: #94a3b8; }
.think-text { color: #86efac; }
.answer-block {
    background: #1a1d27; border: 1px solid #3d4268; border-radius: 8px;
    padding: 12px; margin-top: 8px;
}
.answer-json { color: #fbbf24; }
.answer-action { font-size: 15px; font-weight: 700; color: #f472b6; margin-bottom: 6px; }

.meta-row {
    display: flex; gap: 10px; flex-wrap: wrap; padding: 8px 14px;
    background: #13161f; border-bottom: 1px solid #1e2130; font-size: 12px; color: #6b7280;
}
.meta-pill {
    background: #1e2130; padding: 2px 8px; border-radius: 10px; color: #a5b4fc;
}

.hl-box { color: #fbbf24; }
.hl-pallet { color: #34d399; }
.hl-key { color: #a5b4fc; }
.hl-val { color: #e2e8f0; }
"""

JS = """
function switchTab(name) {
    document.querySelectorAll('.tab').forEach(t => t.classList.toggle('active', t.dataset.tab === name));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.toggle('active', c.id === 'tab-' + name));
}
function jump() {
    const v = parseInt(document.getElementById('jump-input').value, 10);
    const max = parseInt(document.getElementById('jump-input').dataset.max, 10);
    if (!isNaN(v) && v >= 1 && v <= max) location.href = '?file=' + encodeURIComponent(document.getElementById('file-sel').value) + '&idx=' + (v - 1);
}
document.addEventListener('keydown', e => {
    if (e.target.tagName === 'INPUT') return;
    if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') { const a = document.getElementById('btn-prev'); if (a && !a.classList.contains('disabled')) location.href = a.href; }
    if (e.key === 'ArrowRight' || e.key === 'ArrowDown') { const a = document.getElementById('btn-next'); if (a && !a.classList.contains('disabled')) location.href = a.href; }
});
"""


_IMAGE_TAG_RE = re.compile(r"<image>([^<]*)</image>")


def strip_image_tags(text: str) -> str:
    return re.sub(r"<image>[^<]*</image>\n?", "", text).strip()


def extract_inline_images(text: str) -> list[str]:
    """Extract image paths from <image>...</image> tags in the user message."""
    return _IMAGE_TAG_RE.findall(text)


def extract_think(text: str) -> str:
    m = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    return m.group(1).strip() if m else ""


def extract_answer(text: str) -> str:
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return m.group(1).strip() if m else text


def type_from_path(img_path: str, metadata_cache: dict[str, dict]) -> str:
    name = Path(img_path).name
    return metadata_cache.get(name, {}).get("type", "")


def colorize_prompt(text: str) -> str:
    """Apply syntax highlighting to prompt text."""
    lines = []
    for line in text.splitlines():
        hl = html.escape(line)
        if re.match(r"^BOX \d+:", line):
            hl = f'<span class="hl-box">{hl}</span>'
        elif re.match(r"^PALLET \d+:", line):
            hl = f'<span class="hl-pallet">{hl}</span>'
        elif re.match(r"^\s+\w[\w\s]*:", line):
            hl = re.sub(r"^(\s+)([\w][\w\s]*)(:)", lambda m: m.group(1) + f'<span class="hl-key">{m.group(2)}</span>' + m.group(3), hl)
        lines.append(hl)
    return "\n".join(lines)


def load_metadata_cache(jsonl_path: Path) -> dict[str, dict]:
    # Try to find a metadata.json sibling to the sim/images dir
    for candidate in jsonl_path.parents:
        m = candidate / "sim" / "images" / "metadata.json"
        if m.exists():
            with open(m) as f:
                entries = json.load(f)
            return {e["image"]: e for e in entries}
    return {}


def build_page(
    samples: list[dict],
    idx: int,
    jsonl_files: list[Path],
    current_file: Path,
    metadata_cache: dict[str, dict],
) -> str:
    s = samples[idx]
    images = s.get("images") or []
    msgs = s.get("messages", [])
    user_msg = next((m["content"] for m in msgs if m["role"] == "user"), "")
    asst_msg = next((m["content"] for m in msgs if m["role"] == "assistant"), "")

    # Fall back to inline <image> tags if images field is absent (v1 format)
    if not images and isinstance(user_msg, str):
        images = extract_inline_images(user_msg)

    prompt_clean = strip_image_tags(user_msg)
    think = extract_think(asst_msg)
    answer_raw = extract_answer(asst_msg)

    # Parse answer JSON for display
    try:
        answer_data = json.loads(answer_raw)
        action = answer_data.get("action", "?")
        answer_pretty = json.dumps(answer_data, indent=2)
    except Exception:
        action = "?"
        answer_pretty = answer_raw

    # File selector options
    file_opts = "\n".join(
        f'<option value="{p.relative_to(REPO_ROOT)}" {"selected" if p == current_file else ""}>'
        f'{p.relative_to(REPO_ROOT)}</option>'
        for p in jsonl_files
    )

    prev_url = f"?file={current_file.relative_to(REPO_ROOT)}&idx={idx - 1}" if idx > 0 else "#"
    next_url = f"?file={current_file.relative_to(REPO_ROOT)}&idx={idx + 1}" if idx < len(samples) - 1 else "#"

    # Images panel
    imgs_html = ""
    if not images:
        imgs_html = '<p style="color:#4b5563;font-size:12px;padding:8px">No images in this sample.</p>'
    for img_path in images:
        name = Path(img_path).name
        meta = metadata_cache.get(name, {})
        box_type = meta.get("type", "")
        weight = meta.get("weight")
        size = meta.get("size")
        label_cls = f"label-{box_type}" if box_type else ""
        size_str = f"{size[0]}x{size[1]}x{size[2]}m" if size else ""
        weight_str = f"{weight} kg" if weight else ""
        imgs_html += f"""
<div class="img-wrap">
  <span>{html.escape(name)}</span>
  <img src="/img/{img_path}" alt="{html.escape(name)}"
       onerror="this.style.display='none';this.nextElementSibling.style.display='block'">
  <div style="display:none;padding:8px;color:#6b7280;font-size:11px;border:1px dashed #2d3148;border-radius:4px">
    Image not found
  </div>
  <div>
    {f'<span class="img-label {label_cls}">{html.escape(box_type)}</span>' if box_type else ''}
    {f'<span class="img-label" style="color:#6b7280">{html.escape(weight_str)}</span>' if weight_str else ''}
    {f'<span class="img-label" style="color:#4b5563">{html.escape(size_str)}</span>' if size_str else ''}
  </div>
</div>"""

    # Meta pills
    meta_pills = "".join([
        f'<span class="meta-pill">episode {html.escape(str(s.get("episode", "?")))}</span>',
        f'<span class="meta-pill">step {html.escape(str(s.get("step", "?")))}</span>',
        f'<span class="meta-pill">id: {html.escape(str(s.get("id", "?")))}</span>',
        f'<span class="meta-pill action">{html.escape(action)}</span>',
    ])

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Training Viewer — {html.escape(s.get("id", str(idx)))}</title>
<style>{CSS}</style>
</head>
<body>

<div class="topbar">
  <h1>Training Viewer</h1>
  <select id="file-sel" onchange="location.href='?file='+encodeURIComponent(this.value)+'&idx=0'">
    {file_opts}
  </select>
  <div class="nav">
    <a id="btn-prev" class="btn {'disabled' if idx == 0 else ''}" href="{prev_url}">&larr;</a>
    <span class="counter">{idx + 1} / {len(samples)}</span>
    <a id="btn-next" class="btn {'disabled' if idx >= len(samples) - 1 else ''}" href="{next_url}">&rarr;</a>
    <input id="jump-input" type="number" min="1" max="{len(samples)}" placeholder="jump to…"
           data-max="{len(samples)}" style="width:90px"
           onkeydown="if(event.key==='Enter') jump()">
  </div>
</div>

<div class="meta-row">{meta_pills}</div>

<div class="main">
  <div class="images-panel">{imgs_html}</div>

  <div class="content-panel">
    <div class="tabs">
      <button class="tab active" data-tab="prompt" onclick="switchTab('prompt')">Prompt</button>
      <button class="tab" data-tab="think" onclick="switchTab('think')">Think</button>
      <button class="tab" data-tab="answer" onclick="switchTab('answer')">Answer</button>
      <button class="tab" data-tab="raw" onclick="switchTab('raw')">Raw JSON</button>
    </div>

    <div id="tab-prompt" class="tab-content active">
      <pre class="prompt-text">{colorize_prompt(prompt_clean)}</pre>
    </div>

    <div id="tab-think" class="tab-content">
      <pre class="think-text">{html.escape(think) if think else '<span style="color:#4b5563">No &lt;think&gt; block found.</span>'}</pre>
    </div>

    <div id="tab-answer" class="tab-content">
      <div class="answer-action">{html.escape(action)}</div>
      <div class="answer-block">
        <pre class="answer-json">{html.escape(answer_pretty)}</pre>
      </div>
    </div>

    <div id="tab-raw" class="tab-content">
      <pre style="color:#6b7280">{html.escape(json.dumps(s, indent=2))}</pre>
    </div>
  </div>
</div>

<script>{JS}</script>
</body>
</html>"""


# ── HTTP handler ──────────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    jsonl_files: list[Path] = []
    _cache: dict[str, tuple[list[dict], dict]] = {}

    def log_message(self, format, *args):
        pass

    def _get_samples(self, path: Path) -> tuple[list[dict], dict]:
        key = str(path)
        if key not in self._cache:
            samples = load_jsonl(path)
            meta = load_metadata_cache(path)
            self._cache[key] = (samples, meta)
        return self._cache[key]

    def do_GET(self):
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)

        # ── Serve images ──────────────────────────────────────────────────
        if parsed.path.startswith("/img/"):
            rel = parsed.path[5:]  # strip /img/
            abs_path = REPO_ROOT / rel
            if not abs_path.is_file():
                self._404()
                return
            mime = mimetypes.guess_type(str(abs_path))[0] or "application/octet-stream"
            data = abs_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        # ── Main page ─────────────────────────────────────────────────────
        if parsed.path not in ("/", ""):
            self._404()
            return

        if not self.jsonl_files:
            self._text("No JSONL files found under training/dataset/*/processed/")
            return

        file_param = qs.get("file", [None])[0]
        if file_param:
            current_file = REPO_ROOT / file_param
        else:
            current_file = self.jsonl_files[0]

        if current_file not in self.jsonl_files:
            current_file = self.jsonl_files[0]

        samples, meta = self._get_samples(current_file)
        if not samples:
            self._text(f"No samples in {current_file}")
            return

        try:
            idx = max(0, min(int(qs.get("idx", [0])[0]), len(samples) - 1))
        except (ValueError, IndexError):
            idx = 0

        body = build_page(samples, idx, self.jsonl_files, current_file, meta)
        encoded = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _404(self):
        self.send_response(404)
        self.end_headers()

    def _text(self, msg: str):
        body = msg.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Training data viewer")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--file", type=str, default=None, help="Specific JSONL to load first")
    args = parser.parse_args()

    jsonl_files = find_jsonl_files()

    if args.file:
        explicit = REPO_ROOT / args.file
        if explicit not in jsonl_files:
            jsonl_files.insert(0, explicit)

    if not jsonl_files:
        print("No JSONL files found.")
        sys.exit(1)

    Handler.jsonl_files = jsonl_files

    print(f"Found {len(jsonl_files)} JSONL file(s):")
    for f in jsonl_files:
        print(f"  {f.relative_to(REPO_ROOT)}")

    print(f"\nOpen: http://localhost:{args.port}")
    print("Navigate with arrow keys. Ctrl+C to quit.\n")

    server = HTTPServer(("", args.port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
