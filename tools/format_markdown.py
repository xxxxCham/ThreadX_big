import re
import sys
from pathlib import Path


def is_heading(line: str) -> bool:
    return bool(re.match(r"^\s{0,3}#{1,6}\s+\S", line))


def heading_text(line: str) -> str:
    return re.sub(r"^\s*#+\s*", "", line).strip().lower()


def is_list_item(line: str) -> bool:
    return bool(re.match(r"^\s*(?:[-*+]\s+|\d+\.\s+).+", line))


def format_markdown(text: str) -> str:
    lines = text.splitlines()

    # Track code fences to avoid modifying inside code blocks
    in_code = False
    fence = None
    # First pass: remove consecutive duplicate headings
    deduped = []
    prev_heading_key = None
    for line in lines:
        l = line.rstrip("\r\n")
        fence_toggle = re.match(r"^\s*(```|~~~)", l)
        if fence_toggle and (not in_code or fence == fence_toggle.group(1)):
            in_code = not in_code
            fence = fence_toggle.group(1)
            deduped.append(l)
            continue

        if not in_code and is_heading(l):
            key = heading_text(l)
            if prev_heading_key == key:
                # skip duplicate consecutive heading
                continue
            prev_heading_key = key
        else:
            if l.strip():
                prev_heading_key = None
        deduped.append(l)

    # Second pass: ensure blank lines around headings and lists
    result: list[str] = []
    i = 0
    in_code = False
    fence = None
    n = len(deduped)
    while i < n:
        line = deduped[i]
        fence_toggle = re.match(r"^\s*(```|~~~)", line)
        if fence_toggle and (not in_code or fence == fence_toggle.group(1)):
            in_code = not in_code
            fence = fence_toggle.group(1)
            result.append(line)
            i += 1
            continue

        if not in_code and is_heading(line):
            # ensure one blank line before (except at start or already blank)
            if result and result[-1].strip() != "":
                result.append("")
            result.append(line)
            # ensure one blank line after heading
            nxt = deduped[i + 1] if i + 1 < n else None
            if nxt is not None and nxt.strip() != "":
                result.append("")
            i += 1
            continue

        if not in_code and is_list_item(line):
            # Start of list block: ensure blank line before
            if result and result[-1].strip() != "":
                result.append("")
            # emit list block
            while i < n and is_list_item(deduped[i]):
                result.append(deduped[i])
                i += 1
            # ensure blank line after block if next non-blank is not end
            if i < n and deduped[i].strip() != "":
                result.append("")
            continue

        result.append(line)
        i += 1

    # Final pass: collapse multiple blank lines into a single blank line
    collapsed: list[str] = []
    for l in result:
        if l.strip() == "":
            if collapsed and collapsed[-1].strip() == "":
                continue
        collapsed.append(l)

    return "\n".join(collapsed) + ("\n" if text.endswith("\n") else "")


def main(path_str: str) -> None:
    path = Path(path_str)
    text = path.read_text(encoding="utf-8", errors="ignore")
    new_text = format_markdown(text)
    if new_text != text:
        path.write_text(new_text, encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tools/format_markdown.py <file.md>")
        sys.exit(2)
    main(sys.argv[1])

