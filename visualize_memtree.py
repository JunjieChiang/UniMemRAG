#!/usr/bin/env python3

'''
python visualize_memtree.py   --kb-path "../benchmark/infoseek/subset/wiki_text/wiki_5k_dict.json"   --doc-id "https://en.wikipedia.org/wiki/California_quail"   --include-images   --emit-json
'''

from __future__ import annotations

import argparse
import html
import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from unimemrag.memory_forest.memory_forest import MemoryTree, build_tree


def _load_kb(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object (dict) at {path}, got {type(data).__name__}.")
    return data


def _safe_slug(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", value.strip())
    cleaned = cleaned.strip("_")
    return cleaned[:80] if cleaned else "memtree"


def _truncate(text: str, max_chars: int) -> Tuple[str, bool]:
    if max_chars <= 0 or len(text) <= max_chars:
        return text, False
    return text[: max_chars - 1].rstrip() + "...", True


def _format_meta(meta: Dict[str, Any], keys: Sequence[str]) -> str:
    parts = []
    for key in keys:
        value = meta.get(key)
        if value is None or value == "":
            continue
        parts.append(f"{key}: {value}")
    return " | ".join(parts)


def _collect_leaves(tree: MemoryTree) -> Tuple[Dict[str, Any], Dict[str, List[Any]]]:
    leaf_by_id: Dict[str, Any] = {}
    leaves_by_parent: Dict[str, List[Any]] = {}
    for leaf in tree.leaves:
        if leaf.leaf_id:
            leaf_by_id[str(leaf.leaf_id)] = leaf
        parent = str(leaf.parent_id)
        leaves_by_parent.setdefault(parent, []).append(leaf)
    return leaf_by_id, leaves_by_parent


def _render_html(
    tree: MemoryTree,
    *,
    max_events: int,
    max_leaves: int,
    max_leaf_chars: int,
    max_preview_chars: int,
    include_images: bool,
) -> str:
    root = tree.root
    root_meta = root.metadata if isinstance(root.metadata, dict) else {}
    leaf_by_id, leaves_by_parent = _collect_leaves(tree)

    def esc(value: Any) -> str:
        return html.escape(str(value)) if value is not None else ""

    title = root.topic or tree.tree_id
    root_meta_line = _format_meta(
        root_meta,
        ["source_url", "num_sections", "num_images"],
    )

    lines: List[str] = [
        "<!doctype html>",
        "<html lang=\"en\">",
        "<head>",
        "<meta charset=\"utf-8\" />",
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />",
        f"<title>{esc(title)}</title>",
        "<style>",
        "  :root {",
        "    --bg: #f7f3ee;",
        "    --ink: #1b1b1b;",
        "    --muted: #6a5f55;",
        "    --accent: #c0482b;",
        "    --card: #fffdf9;",
        "    --border: #e2d8cd;",
        "  }",
        "  body {",
        "    margin: 0;",
        "    padding: 32px;",
        "    font-family: \"Iowan Old Style\", \"Georgia\", serif;",
        "    background: radial-gradient(circle at top, #fffaf3, var(--bg));",
        "    color: var(--ink);",
        "  }",
        "  .header {",
        "    margin-bottom: 24px;",
        "  }",
        "  h1 {",
        "    margin: 0 0 8px;",
        "    font-size: 28px;",
        "    letter-spacing: 0.2px;",
        "  }",
        "  .meta {",
        "    color: var(--muted);",
        "    font-size: 14px;",
        "  }",
        "  .tree {",
        "    display: grid;",
        "    gap: 16px;",
        "  }",
        "  details {",
        "    background: var(--card);",
        "    border: 1px solid var(--border);",
        "    border-radius: 12px;",
        "    padding: 12px 14px;",
        "  }",
        "  summary {",
        "    cursor: pointer;",
        "    font-weight: 600;",
        "  }",
        "  .event-meta {",
        "    margin-top: 6px;",
        "    color: var(--muted);",
        "    font-size: 13px;",
        "  }",
        "  .summary {",
        "    margin: 10px 0;",
        "    padding-left: 12px;",
        "    border-left: 3px solid var(--accent);",
        "  }",
        "  .leaves {",
        "    display: grid;",
        "    gap: 10px;",
        "    margin-top: 10px;",
        "  }",
        "  .leaf {",
        "    padding: 10px 12px;",
        "    background: #fff8ee;",
        "    border-radius: 10px;",
        "    border: 1px dashed #e4c9b7;",
        "    font-size: 14px;",
        "    line-height: 1.5;",
        "  }",
        "  .leaf-meta {",
        "    color: var(--muted);",
        "    font-size: 12px;",
        "    margin-bottom: 6px;",
        "  }",
        "  img.root-image {",
        "    max-width: 320px;",
        "    border-radius: 12px;",
        "    border: 1px solid var(--border);",
        "    margin-top: 12px;",
        "  }",
        "</style>",
        "</head>",
        "<body>",
        "<div class=\"header\">",
        f"<h1>{esc(title)}</h1>",
        f"<div class=\"meta\">Tree ID: {esc(tree.tree_id)}"
        + (f" | {esc(root_meta_line)}" if root_meta_line else "")
        + "</div>",
    ]

    if include_images and root.image_uri:
        lines.extend(
            [
                f"<img class=\"root-image\" src=\"{esc(root.image_uri)}\" alt=\"root image\" />",
            ]
        )

    lines.append("</div>")
    lines.append("<div class=\"tree\">")

    for event in tree.events[:max_events]:
        emeta = event.metadata if isinstance(event.metadata, dict) else {}
        title_value = emeta.get("section_title") or event.summary or f"Event {event.event_id}"
        preview = emeta.get("section_preview") or ""
        preview_text, preview_trimmed = _truncate(str(preview), max_preview_chars)
        summary_text, _ = _truncate(str(event.summary or ""), max_preview_chars)
        section_meta = _format_meta(
            emeta,
            ["section_index", "section_title"],
        )

        lines.append("<details open>")
        lines.append(f"<summary>{esc(title_value)}</summary>")
        if section_meta:
            lines.append(f"<div class=\"event-meta\">{esc(section_meta)}</div>")
        if summary_text:
            lines.append(f"<div class=\"summary\"><strong>Summary:</strong> {esc(summary_text)}</div>")
        if preview_text and preview_text != summary_text:
            suffix = " (trimmed)" if preview_trimmed else ""
            lines.append(f"<div class=\"summary\"><strong>Preview{suffix}:</strong> {esc(preview_text)}</div>")

        leaf_nodes: List[Any] = []
        if event.leaf_ids:
            for leaf_id in event.leaf_ids:
                leaf = leaf_by_id.get(str(leaf_id))
                if leaf is not None:
                    leaf_nodes.append(leaf)
        if not leaf_nodes:
            leaf_nodes = list(leaves_by_parent.get(str(event.event_id), []))

        if max_leaves > 0:
            leaf_nodes = leaf_nodes[:max_leaves]

        if leaf_nodes:
            lines.append("<div class=\"leaves\">")
            for idx, leaf in enumerate(leaf_nodes, start=1):
                leaf_meta = leaf.metadata if isinstance(leaf.metadata, dict) else {}
                leaf_meta_line = _format_meta(
                    leaf_meta,
                    ["paragraph_index", "section_title"],
                )
                leaf_text, trimmed = _truncate(str(leaf.text or ""), max_leaf_chars)
                suffix = " (trimmed)" if trimmed else ""
                lines.append("<div class=\"leaf\">")
                lines.append(f"<div class=\"leaf-meta\">Leaf {idx}{suffix}</div>")
                if leaf_meta_line:
                    lines.append(f"<div class=\"leaf-meta\">{esc(leaf_meta_line)}</div>")
                lines.append(f"<div>{esc(leaf_text)}</div>")
                lines.append("</div>")
            lines.append("</div>")

        lines.append("</details>")

    lines.append("</div>")
    lines.append("</body>")
    lines.append("</html>")

    return "\n".join(lines)


def _tree_to_json(tree: MemoryTree) -> str:
    return json.dumps(asdict(tree), ensure_ascii=False, indent=2)


def _select_doc(kb: Dict[str, Any], doc_id: str) -> Tuple[str, Dict[str, Any]]:
    if doc_id in kb:
        return doc_id, kb[doc_id]
    suggestions = [key for key in kb.keys() if doc_id in key]
    hint = f" Did you mean one of: {', '.join(suggestions[:5])}?" if suggestions else ""
    raise KeyError(f"doc_id {doc_id!r} not found.{hint}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize a single MemoryTree from a wiki dict JSON.")
    parser.add_argument("--kb-path", type=Path, required=True, help="Path to wiki_*_dict*.json")
    parser.add_argument("--doc-id", type=str, required=True, help="Document id / wiki URL key")
    parser.add_argument("--output", type=Path, default=None, help="Output HTML path")
    parser.add_argument("--chunk-size", type=int, default=1024, help="Chunk size used for leaf nodes")
    parser.add_argument("--chunk-overlap", type=int, default=120, help="Chunk overlap used for leaf nodes")
    parser.add_argument("--max-events", type=int, default=30, help="Max event nodes to show")
    parser.add_argument("--max-leaves", type=int, default=4, help="Max leaf nodes per event")
    parser.add_argument("--max-leaf-chars", type=int, default=260, help="Max characters per leaf")
    parser.add_argument("--max-preview-chars", type=int, default=280, help="Max characters per summary/preview")
    parser.add_argument("--include-images", action="store_true", help="Show root image if present")
    parser.add_argument("--emit-json", action="store_true", help="Also write the raw tree JSON next to HTML")

    args = parser.parse_args()

    kb = _load_kb(args.kb_path)
    doc_id, payload = _select_doc(kb, args.doc_id)
    tree = build_tree(
        doc_id,
        payload,
        chunk_size=max(1, args.chunk_size),
        chunk_overlap=max(0, args.chunk_overlap),
        show_progress=False,
    )

    output_path = args.output
    if output_path is None:
        slug = _safe_slug(doc_id)
        output_path = Path(f"memtree_{slug}.html")

    html_text = _render_html(
        tree,
        max_events=max(0, args.max_events),
        max_leaves=max(0, args.max_leaves),
        max_leaf_chars=max(0, args.max_leaf_chars),
        max_preview_chars=max(0, args.max_preview_chars),
        include_images=bool(args.include_images),
    )
    output_path.write_text(html_text, encoding="utf-8")

    if args.emit_json:
        json_path = output_path.with_suffix(".json")
        json_path.write_text(_tree_to_json(tree), encoding="utf-8")

    print(f"Wrote: {output_path}")
    if args.emit_json:
        print(f"Wrote: {output_path.with_suffix('.json')}")


if __name__ == "__main__":
    main()
