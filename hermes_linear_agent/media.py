"""Media attached to a Linear Agent Chat prompt.

``AgentActivityPromptContent`` (Linear's public schema) carries only ``body``
(markdown), ``bodyData`` (internal ProseMirror JSON) and ``title`` — there is no
attachment field. Screenshots and files therefore arrive, when Linear sends them
at all, as rich nodes inside ``bodyData`` and occasionally as markdown image
links inside ``body``.

This module turns those references into locally cached files so the gateway's
normal vision path can read them, and can capture the raw prompt payload so an
operator can prove what Linear actually delivered for a given attachment.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# ProseMirror node types that mean "this node is an attachment, not prose".
MEDIA_NODE_TYPES = frozenset({"image", "file", "attachment", "upload", "video", "audio"})

# Attributes a rich node may carry the asset URL in.
_URL_ATTRS = ("src", "href", "url", "downloadUrl", "assetUrl", "fileUrl")

_MEDIA_EXTENSIONS = frozenset(
    {
        # images
        ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tif", ".tiff", ".heic", ".avif", ".svg",
        # documents
        ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".csv", ".ppt", ".pptx", ".txt", ".rtf", ".odt",
        # archives / other
        ".zip", ".tar", ".gz", ".7z",
        # video / audio
        ".mp4", ".mov", ".webm", ".mkv", ".avi", ".mp3", ".wav", ".m4a", ".ogg",
    }
)

# Linear serves chat uploads from its own CDN; those may need the app token.
_LINEAR_UPLOAD_HOSTS = ("uploads.linear.app", "linear.app")

_MARKDOWN_IMAGE_RE = re.compile(r"!\[[^\]]*\]\(\s*<?([^)\s>]+)>?")
# Linear serializes an attachment node into the *body text* of a prompt as a
# custom tag, e.g.:
#   <linear-image>{"type":"image","attrs":{"src":"[https://uploads.linear.app/…]",
#   "title":"image.png","width":1331,"height":432}}</linear-image>
# Note the URL is wrapped in square brackets inside the JSON.
_LINEAR_TAG_RE = re.compile(
    r"<linear-(?P<kind>[a-z_]+)>\s*(?P<payload>\{.*?\})\s*</linear-(?P=kind)>", re.DOTALL
)
_KIND_MEDIA_HINT = {"image", "file", "attachment", "upload", "video", "audio", "document"}
# Tight termination: JSON/markdown that surrounds a URL must not be swallowed.
_BARE_URL_RE = re.compile(r"https?://[^\s\"'<>()\[\]{},;]+")
MAX_URL_CHARS = 4096

MAX_ITEMS = 10
MAX_BYTES = 25 * 1024 * 1024
DOWNLOAD_TIMEOUT_SECONDS = 30.0

CAPTURE_ENV = "LINEAR_AGENT_CAPTURE_PROMPTS"
CAPTURE_MAX_CHARS = 200_000


@dataclass(frozen=True)
class MediaReference:
    """One asset referenced by a prompt."""

    url: str
    name: str = ""
    mime: str = ""
    source: str = "bodyData"


def _extension(url: str) -> str:
    path = urlparse(url).path or ""
    _, _, ext = path.rpartition(".")
    return f".{ext.lower()}" if ext else ""


def is_media_url(url: str) -> bool:
    """True when *url* plausibly points at an attachment we should fetch.

    Issue/profile links and other in-app navigation are excluded; anything on
    Linear's upload CDN or carrying a known media extension is included.
    """
    if not isinstance(url, str):
        return False
    candidate = url.strip()
    if not candidate.startswith(("http://", "https://")):
        return False
    parsed = urlparse(candidate)
    host = (parsed.netloc or "").lower()
    if "uploads.linear.app" in host:
        return True
    if _extension(candidate) in _MEDIA_EXTENSIONS:
        # A Linear issue/profile URL can still end in a document-like slug; only
        # treat linear.app links as media when they carry a media extension.
        return True
    return False


def _filename_for(url: str) -> str:
    name = os.path.basename(urlparse(url).path or "")
    return name or "linear-attachment"


def _collect_from_body_data(node: Any, found: list[MediaReference]) -> None:
    """Walk a ProseMirror document for attachment nodes and asset URLs."""
    if isinstance(node, dict):
        node_type = str(node.get("type") or "").lower()
        attrs = node.get("attrs") if isinstance(node.get("attrs"), dict) else {}
        marks = node.get("marks") if isinstance(node.get("marks"), list) else []
        is_media_node = node_type in MEDIA_NODE_TYPES
        candidate_urls: list[str] = []
        for attr in _URL_ATTRS:
            value = node.get(attr)
            if value is None and isinstance(attrs, dict):
                value = attrs.get(attr)
            if isinstance(value, str) and value.strip():
                candidate_urls.append(value.strip())
        for mark in marks:
            if isinstance(mark, dict):
                attrs_m = mark.get("attrs") if isinstance(mark.get("attrs"), dict) else {}
                for attr in _URL_ATTRS:
                    value = attrs_m.get(attr)
                    if isinstance(value, str) and value.strip():
                        candidate_urls.append(value.strip())
        name = ""
        for key in ("filename", "fileName", "name", "title"):
            value = attrs.get(key) if isinstance(attrs, dict) else None
            if isinstance(value, str) and value.strip():
                name = value.strip()
                break
        for url in candidate_urls:
            if is_media_url(url) or is_media_node:
                found.append(MediaReference(url=url, name=name or _filename_for(url)))
        for key, value in node.items():
            if key in _URL_ATTRS or key == "attrs":
                if key == "attrs" and isinstance(value, dict):
                    _collect_from_body_data(value, found)
                continue
            _collect_from_body_data(value, found)
    elif isinstance(node, list):
        for item in node:
            _collect_from_body_data(item, found)


def _strip_url_wrappers(value: str) -> str:
    """Undo Linear's own rendering quirks around an asset URL.

    The ``<linear-image>`` tag carries the URL inside ``[brackets]``, but the
    closing bracket is not always present in what Linear serializes, so strip
    unmatched wrappers on either side rather than requiring a pair.
    """
    text = value.strip()
    text = text.strip("[]<>").strip()
    return text.rstrip(".,;").strip()


def _collect_from_linear_tags(body: str, found: list[MediaReference]) -> None:
    """Parse Linear's ``<linear-image>{json}</linear-image>`` prompt tags."""
    for match in _LINEAR_TAG_RE.finditer(body or ""):
        kind = (match.group("kind") or "").lower()
        try:
            node = json.loads(match.group("payload"))
        except (ValueError, TypeError):
            continue
        if not isinstance(node, dict):
            continue
        attrs = node.get("attrs") if isinstance(node.get("attrs"), dict) else {}
        name = ""
        for key in ("title", "filename", "fileName", "name"):
            value = attrs.get(key) or node.get(key)
            if isinstance(value, str) and value.strip():
                name = value.strip()
                break
        for attr in _URL_ATTRS:
            value = attrs.get(attr) or node.get(attr)
            if not isinstance(value, str) or not value.strip():
                continue
            url = _strip_url_wrappers(value)
            if not url or len(url) > MAX_URL_CHARS:
                continue
            is_media_node = kind in _KIND_MEDIA_HINT or str(node.get("type") or "").lower() in _KIND_MEDIA_HINT
            if is_media_url(url) or is_media_node:
                found.append(MediaReference(url=url, name=name or _filename_for(url), source="tag"))


def _collect_from_body(body: str, found: list[MediaReference]) -> None:
    if not body:
        return
    _collect_from_linear_tags(body, found)
    for match in _MARKDOWN_IMAGE_RE.finditer(body):
        url = _strip_url_wrappers(match.group(1))
        if is_media_url(url):
            found.append(MediaReference(url=url, name=_filename_for(url), source="body"))
    for match in _BARE_URL_RE.finditer(body):
        url = match.group(0).rstrip(".,;")
        if len(url) > MAX_URL_CHARS:
            continue
        if is_media_url(url):
            found.append(MediaReference(url=url, name=_filename_for(url), source="body"))


def _dedupe_key(url: str) -> str:
    """Identity of an asset, ignoring Linear's rotating signature parameter."""
    parsed = urlparse(url)
    if "uploads.linear.app" in (parsed.netloc or "").lower():
        return f"{parsed.netloc}{parsed.path}"
    return url


def extract_media_references(body: str | None, body_data: Any) -> list[MediaReference]:
    """Media referenced by a prompt, de-duplicated, in discovery order.

    A prompt can mention the same upload twice (Linear's tag plus a markdown
    copy); each signature query differs, so identity is host+path for Linear
    uploads.
    """
    found: list[MediaReference] = []
    _collect_from_body_data(body_data, found)
    _collect_from_body(body or "", found)
    unique: list[MediaReference] = []
    seen: set[str] = set()
    for ref in found:
        key = _dedupe_key(ref.url)
        if key in seen:
            continue
        seen.add(key)
        unique.append(ref)
    return unique


async def download_media(
    references: Iterable[MediaReference],
    *,
    access_token: str = "",
    max_items: int = MAX_ITEMS,
    max_bytes: int = MAX_BYTES,
) -> list[tuple[str, str]]:
    """Download references into the local media cache; return [(path, mime)].

    Failures are logged and skipped: an unreadable attachment must never break
    the user's turn.
    """
    import aiohttp  # local import keeps module import light for pure tests

    from gateway.platforms.base import cache_media_bytes_async

    results: list[tuple[str, str]] = []
    for reference in list(references)[:max_items]:
        url = reference.url
        if not url.startswith(("http://", "https://")):
            continue
        headers = {}
        host = (urlparse(url).netloc or "").lower()
        if access_token and any(h in host for h in _LINEAR_UPLOAD_HOSTS):
            headers["Authorization"] = f"Bearer {access_token}"
        try:
            timeout = aiohttp.ClientTimeout(total=DOWNLOAD_TIMEOUT_SECONDS)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, headers=headers, allow_redirects=True) as response:
                    if response.status >= 400:
                        logger.warning(
                            "[linear_agent] Attachment fetch failed (%s) for %s",
                            response.status,
                            url,
                        )
                        continue
                    if response.content_length and response.content_length > max_bytes:
                        logger.warning("[linear_agent] Attachment too large, skipping %s", url)
                        continue
                    # aiohttp's content.read(n) can return a short body for these
                    # CDN responses (observed 4.5 KB of a 37 KB PNG on 3.14.3),
                    # which produced a corrupt image in the cache. Consume the
                    # stream explicitly instead.
                    chunks: list[bytes] = []
                    total = 0
                    async for chunk in response.content.iter_chunked(65536):
                        total += len(chunk)
                        if total > max_bytes:
                            break
                        chunks.append(chunk)
                    data = b"".join(chunks)
                    content_type = str(response.headers.get("Content-Type") or "").split(";")[0].strip()
            if not data or len(data) > max_bytes:
                logger.warning("[linear_agent] Attachment empty or oversized, skipping %s", url)
                continue
            cached = await cache_media_bytes_async(
                data,
                filename=reference.name or _filename_for(url),
                mime_type=reference.mime or content_type,
            )
            if not cached:
                logger.warning("[linear_agent] Attachment rejected by media validation: %s", url)
                continue
            results.append((str(cached.path), str(cached.media_type)))
            logger.info("[linear_agent] Ingested prompt attachment %s (%s)", cached.path, cached.media_type)
        except Exception as exc:  # noqa: BLE001 - attachment failure must not break the turn
            logger.warning("[linear_agent] Attachment ingest error for %s: %s", url, exc)
    return results


def activity_content_from_payload(payload: Any) -> tuple[str, Any]:
    """Return (body, bodyData) of the prompting activity inside a webhook payload."""
    if not isinstance(payload, dict):
        return "", None
    activity = payload.get("agentActivity")
    if not isinstance(activity, dict):
        data = payload.get("data")
        activity = data.get("agentActivity") if isinstance(data, dict) else None
    if not isinstance(activity, dict):
        return "", None
    content = activity.get("content")
    if isinstance(content, str):
        return content, None
    if not isinstance(content, dict):
        return "", None
    body = content.get("body")
    body_data = content.get("bodyData")
    return (body if isinstance(body, str) else ""), body_data


def captures_enabled() -> bool:
    return str(os.getenv(CAPTURE_ENV, "")).strip().lower() in {"1", "true", "yes", "on"}


def capture_path() -> Path:
    from hermes_constants import get_hermes_home

    try:
        home = get_hermes_home()
    except Exception:  # noqa: BLE001 - fall back to the default home
        home = Path.home() / ".hermes"
    return Path(home) / "plugin-state" / "linear-agent" / "prompt-captures.jsonl"


def capture_prompt_payload(payload: Any, *, action: str = "", session_id: str = "") -> Path | None:
    """Append the raw prompt payload (bounded) for attachment diagnosis.

    Enabled with ``LINEAR_AGENT_CAPTURE_PROMPTS=1``. The file lives in
    profile-scoped plugin state with ``0600`` permissions because it contains
    customer text, and is truncated so a hostile payload cannot exhaust disk.
    """
    if not captures_enabled():
        return None
    try:
        path = capture_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "captured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "action": action,
            "agent_session_id": session_id,
            "payload": payload,
        }
        serialized = json.dumps(record, ensure_ascii=False, default=str)
        if len(serialized) > CAPTURE_MAX_CHARS:
            serialized = serialized[:CAPTURE_MAX_CHARS] + '"}]}'
        with path.open("a", encoding="utf-8") as handle:
            os.chmod(path, 0o600)
            handle.write(serialized + "\n")
        return path
    except Exception as exc:  # noqa: BLE001 - diagnosis must never break dispatch
        logger.warning("[linear_agent] Prompt capture failed: %s", exc)
        return None
