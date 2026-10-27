"""HTML sanitiser: remove active content and, above all, hidden text.

Hidden text is the main carrier of indirect prompt injection: the page looks
normal to a human while the scraper feeds the model instructions nobody sees.
This removes elements a browser would not show (inline-style hiding, ``hidden``,
``aria-hidden``, off-screen positioning, zero-size fonts, white-on-white, common
"visually hidden" classes) plus scripts, frames, forms, comments and metadata.

Limit: only *inline* styles and well-known class names are understood. Text
hidden by an external stylesheet cannot be seen here, which is why the injection
screen and the generator hardening exist as further layers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from bs4 import BeautifulSoup, Comment, Declaration, Doctype, ProcessingInstruction, Tag

from offline_pipeline.safety.errors import UnsafeContent

MAX_HTML_CHARS = 10_000_000

_REMOVE_TAGS = {
    "script", "style", "noscript", "iframe", "frame", "frameset", "object", "embed",
    "applet", "form", "input", "button", "select", "textarea", "svg", "canvas",
    "template", "meta", "link", "base", "audio", "video", "source", "track", "map",
    "nav", "footer", "header", "aside",
}
_HIDDEN_CLASSES = {
    "sr-only", "visually-hidden", "visuallyhidden", "screen-reader-text", "screenreader",
    "hidden", "d-none", "hide", "invisible", "offscreen", "off-screen", "assistive-text",
}
_NAMED_COLORS_WHITE = {"white", "#fff", "#ffffff", "rgb(255,255,255)", "rgba(255,255,255,1)"}


@dataclass
class SanitizeReport:
    hidden_elements_removed: int = 0
    active_elements_removed: int = 0


def _parse_style(style: str) -> dict[str, str]:
    props: dict[str, str] = {}
    for declaration in style.split(";"):
        if ":" in declaration:
            key, _, value = declaration.partition(":")
            props[key.strip().lower()] = re.sub(r"\s+", "", value.strip().lower().replace("!important", ""))
    return props


def _px(value: str) -> float | None:
    match = re.fullmatch(r"(-?\d+(?:\.\d+)?)(px|pt|em|rem|%)?", value)
    if not match:
        return None
    number, unit = float(match.group(1)), match.group(2) or "px"
    return {"px": number, "pt": number * 1.33, "em": number * 16, "rem": number * 16, "%": number * 0.16}[unit]


def _is_white(value: str | None) -> bool:
    if not value:
        return False
    if value in _NAMED_COLORS_WHITE:
        return True
    match = re.fullmatch(r"rgba?\((\d+),(\d+),(\d+)(?:,[\d.]+)?\)", value)
    return bool(match) and all(int(g) >= 245 for g in match.groups())


def style_hides(style: str) -> bool:
    """True when an inline style makes an element invisible to a human reader."""
    p = _parse_style(style)
    if p.get("display") == "none" or p.get("visibility") in {"hidden", "collapse"}:
        return True
    try:
        if float(p.get("opacity", "1")) <= 0.05:
            return True
    except ValueError:
        pass
    size = _px(p["font-size"]) if "font-size" in p else None
    if size is not None and size <= 2:
        return True
    for dimension in ("height", "width", "max-height", "max-width"):
        if p.get(dimension) in {"0", "0px", "0pt"} and p.get("overflow") in {"hidden", "clip"}:
            return True
    indent = _px(p["text-indent"]) if "text-indent" in p else None
    if indent is not None and indent <= -999:
        return True
    for edge in ("left", "right", "top", "bottom"):
        offset = _px(p[edge]) if edge in p else None
        if offset is not None and offset <= -999:
            return True
    if p.get("clip", "").startswith("rect(0") or p.get("clip-path", "").startswith("inset(50%"):
        return True
    color = p.get("color")
    background = p.get("background-color") or p.get("background")
    if color == "transparent":
        return True
    if _is_white(color) and (not background or _is_white(background)):
        return True
    return bool(color and background and color == background)


def _is_hidden(tag: Tag) -> bool:
    attrs = tag.attrs or {}
    if "hidden" in attrs or str(attrs.get("aria-hidden", "")).lower() == "true":
        return True
    classes = attrs.get("class") or []
    if isinstance(classes, str):
        classes = classes.split()
    if any(c.lower() in _HIDDEN_CLASSES for c in classes):
        return True
    return style_hides(str(attrs.get("style", "")))


def sanitize_html(html: str) -> tuple[BeautifulSoup, SanitizeReport]:
    """Parse *html* and return a sanitised tree plus counts of what was removed."""
    if len(html) > MAX_HTML_CHARS:
        raise UnsafeContent("html_too_large", "HTML document is too large.")
    report = SanitizeReport()
    try:
        soup = BeautifulSoup(html, "html.parser")
        for node in soup.find_all(string=lambda s: isinstance(s, (Comment, Declaration, Doctype, ProcessingInstruction))):
            node.extract()
        for tag in list(soup.find_all(True)):
            if tag.decomposed:  # already removed with an ancestor
                continue
            name = (tag.name or "").lower()
            if name == "body" or name == "html":
                continue
            if name in _REMOVE_TAGS:
                tag.decompose()
                report.active_elements_removed += 1
            elif _is_hidden(tag):
                tag.decompose()
                report.hidden_elements_removed += 1
    except RecursionError as exc:
        raise UnsafeContent("html_too_deep", "HTML is nested too deeply.") from exc
    return soup, report


def html_to_text(soup: BeautifulSoup) -> str:
    try:
        return soup.get_text(separator="\n", strip=True)
    except RecursionError as exc:
        raise UnsafeContent("html_too_deep", "HTML is nested too deeply.") from exc
