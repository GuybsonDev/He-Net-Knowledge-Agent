import re
from urllib.parse import urlparse

from bs4 import BeautifulSoup, Tag

from henet_kb.ingest.base import Document, Section

DROP_TAGS = (
    "script",
    "style",
    "noscript",
    "svg",
    "iframe",
    "form",
    "button",
    "input",
    "select",
    "textarea",
    "header",
    "footer",
    "nav",
    "template",
)

# Elementor and JetPlugins put menus, popups and the cookie banner under these classes.
DROP_CLASS_RE = re.compile(
    r"cookie|popup|jet-mobile-menu|jet-menu|elementor-location-header|"
    r"elementor-location-footer|breadcrumb|share|screen-reader|skip-link|menu-item",
    re.IGNORECASE,
)

HEADING_TAGS = ("h1", "h2", "h3")
TEXT_TAGS = ("h4", "h5", "h6", "p", "li", "td", "th", "blockquote", "pre", "dd", "dt")
TITLE_SUFFIX_RE = re.compile(r"\s+[-|]\s+He-?Net.*$", re.IGNORECASE)
WHITESPACE_RE = re.compile(r"\s+")


def normalize_whitespace(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text).strip()


def _strip_chrome(root: Tag) -> None:
    for tag in root.find_all(DROP_TAGS):
        tag.decompose()
    for tag in root.find_all(True):
        # Children of an element removed earlier in this loop lose their attrs.
        if tag.decomposed or tag.attrs is None:
            continue
        classes = tag.get("class") or []
        if isinstance(classes, str):
            classes = [classes]
        marker = " ".join(classes) + " " + str(tag.get("id") or "")
        if DROP_CLASS_RE.search(marker):
            tag.decompose()


def _content_root(soup: BeautifulSoup) -> Tag:
    for selector in ("main", "article", ".entry-content", "body"):
        found = soup.select_one(selector)
        if found is not None:
            return found
    return soup


def html_to_sections(html: str) -> list[Section]:
    """Turn an HTML page or fragment into plain text sections, one per heading."""
    soup = BeautifulSoup(html, "lxml")
    root = _content_root(soup)
    _strip_chrome(root)

    sections: list[Section] = []
    heading = ""
    buffer: list[str] = []

    def flush() -> None:
        text = normalize_whitespace(" ".join(buffer))
        if text:
            sections.append(Section(heading=heading, text=text))
        buffer.clear()

    for tag in root.find_all(HEADING_TAGS + TEXT_TAGS):
        # A p inside an li would otherwise be emitted twice.
        if tag.find_parent(TEXT_TAGS) is not None:
            continue
        text = normalize_whitespace(tag.get_text(" "))
        if not text:
            continue
        if tag.name in HEADING_TAGS:
            flush()
            heading = text
            buffer.append(text)
        else:
            buffer.append(text)
    flush()

    # Some pages are built from bare divs and have almost no block tags. Use the raw
    # text only when it clearly holds more than the structured pass found.
    total = sum(len(section.text) for section in sections)
    if total < 200:
        fallback = normalize_whitespace(root.get_text(" "))
        if len(fallback) > 2 * total:
            sections = [Section(heading="", text=fallback)] if fallback else []
    return sections


def extract_title(html: str, url: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    og = soup.find("meta", property="og:title")
    candidates = [
        og.get("content") if isinstance(og, Tag) else None,
        soup.title.get_text() if soup.title else None,
        soup.h1.get_text() if soup.h1 else None,
    ]
    for candidate in candidates:
        if candidate and normalize_whitespace(candidate):
            return TITLE_SUFFIX_RE.sub("", normalize_whitespace(candidate))
    path = urlparse(url).path.strip("/")
    return path.split("/")[-1].replace("-", " ").capitalize() if path else "Home"


def html_to_document(
    html: str, url: str, source: str = "", title: str | None = None, modified: str | None = None
) -> Document:
    return Document(
        url=url,
        title=title or extract_title(html, url),
        sections=html_to_sections(html),
        source=source,
        modified=modified,
    )
