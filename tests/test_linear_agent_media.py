"""Invariant tests for Linear Agent prompt-attachment ingestion (media.py)."""

from __future__ import annotations

import json
import os
import stat

import pytest

from hermes_linear_agent import media


def test_extracts_prosemirror_image_node():
    body_data = {
        "type": "doc",
        "content": [
            {"type": "paragraph", "content": [{"type": "text", "text": "see screenshot"}]},
            {
                "type": "image",
                "attrs": {"src": "https://uploads.linear.app/abc/screenshot.png", "alt": "screenshot"},
            },
        ],
    }
    refs = media.extract_media_references("see screenshot", body_data)
    assert [r.url for r in refs] == ["https://uploads.linear.app/abc/screenshot.png"]
    assert refs[0].name == "screenshot.png"


def test_extracts_markdown_image_and_bare_upload_url_from_body():
    body = (
        "Two shots: ![first](https://uploads.linear.app/a/one.png) and "
        "plain https://uploads.linear.app/b/two.pdf here."
    )
    urls = [r.url for r in media.extract_media_references(body, None)]
    assert "https://uploads.linear.app/a/one.png" in urls
    assert "https://uploads.linear.app/b/two.pdf" in urls


def test_ignores_prose_links_that_are_not_media():
    body = (
        "See https://linear.app/casapay/issue/SUP-123/some-title and "
        "https://app.hubspot.com/contacts/148483107/ticket/1"
    )
    assert media.extract_media_references(body, None) == []


def test_deduplicates_same_asset_from_body_and_body_data():
    url = "https://uploads.linear.app/x/shot.jpg"
    refs = media.extract_media_references(
        f"![shot]({url})", {"type": "image", "attrs": {"src": url}}
    )
    assert [r.url for r in refs] == [url]


def test_extracts_linear_image_tag_with_bracketed_src():
    """The real Linear shape: a <linear-image> tag with the URL in [brackets]."""
    body = (
        "why is the collection missing?\n\n"
        '<linear-image>{"type":"image","attrs":{"src":"[https://uploads.linear.app/ws/thread/'
        'e0554059-1061-4c28-bce0-e99ac14820d9?signature=eyJhbGciOi","title":"image.png",'
        '"width":1331,"height":432}}</linear-image>'
    )
    refs = media.extract_media_references(body, None)
    assert len(refs) == 1
    assert refs[0].url == (
        "https://uploads.linear.app/ws/thread/e0554059-1061-4c28-bce0-e99ac14820d9?signature=eyJhbGciOi"
    )
    assert refs[0].name == "image.png"


def test_bare_url_does_not_swallow_surrounding_json():
    body = 'see https://uploads.linear.app/a/b.png?signature=abc","title":"image.png","width":1331'
    urls = [r.url for r in media.extract_media_references(body, None)]
    assert urls == ["https://uploads.linear.app/a/b.png?signature=abc"]


def test_same_upload_with_different_signatures_is_one_reference():
    body = (
        '<linear-image>{"type":"image","attrs":{"src":"[https://uploads.linear.app/a/b.png?sig=1]"}}'
        "</linear-image>\n![shot](https://uploads.linear.app/a/b.png?sig=2)"
    )
    refs = media.extract_media_references(body, None)
    assert len(refs) == 1


def test_text_only_prompt_yields_no_media():
    assert media.extract_media_references("please make a draft", {"type": "doc", "content": []}) == []


def test_capture_is_off_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv(media.CAPTURE_ENV, raising=False)
    monkeypatch.setattr(media, "capture_path", lambda: tmp_path / "prompt-captures.jsonl")
    assert media.capture_prompt_payload({"a": 1}, action="prompted") is None
    assert not (tmp_path / "prompt-captures.jsonl").exists()


def test_capture_writes_private_jsonl_when_enabled(tmp_path, monkeypatch):
    monkeypatch.setenv(media.CAPTURE_ENV, "1")
    target = tmp_path / "prompt-captures.jsonl"
    monkeypatch.setattr(media, "capture_path", lambda: target)
    payload = {"action": "prompted", "agentActivity": {"content": {"bodyData": {"type": "doc"}}}}
    assert media.capture_prompt_payload(payload, action="prompted", session_id="s-1") == target
    lines = target.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["payload"]["agentActivity"]["content"]["bodyData"]["type"] == "doc"
    assert record["agent_session_id"] == "s-1"
    assert stat.S_IMODE(os.stat(target).st_mode) == 0o600


@pytest.mark.asyncio
async def test_download_skips_failures_without_raising(monkeypatch):
    """A 404 attachment must be skipped; a good one must land in the media cache."""
    fetched: list[str] = []

    class FakeResponse:
        def __init__(self, status, data=b"", content_type="image/png"):
            self.status = status
            self._data = data
            self.headers = {"Content-Type": content_type}
            self.content_length = len(data)
            self.content = self

        async def read(self, _limit):
            return self._data

        async def iter_chunked(self, _size):
            for i in range(0, len(self._data), 4) or [0]:
                yield self._data[i : i + 4]

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    class FakeSession:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        def get(self, url, **kw):
            fetched.append(url)
            if url.endswith("missing.png"):
                return FakeResponse(404)
            return FakeResponse(200, b"\x89PNG\r\n\x1a\n" + b"0" * 16)

    import aiohttp

    monkeypatch.setattr(aiohttp, "ClientSession", FakeSession)
    monkeypatch.setattr(aiohttp, "ClientTimeout", lambda **kw: None)

    class Cached:
        def __init__(self):
            self.path = "/tmp/cached-image.png"
            self.media_type = "image/png"

    async def fake_cache(data, *, filename="", mime_type="", default_kind=None):
        return Cached()

    monkeypatch.setattr("gateway.platforms.base.cache_media_bytes_async", fake_cache, raising=False)

    refs = [
        media.MediaReference(url="https://uploads.linear.app/a/missing.png"),
        media.MediaReference(url="https://uploads.linear.app/a/good.png"),
    ]
    out = await media.download_media(refs, access_token="tok")
    assert fetched == [
        "https://uploads.linear.app/a/missing.png",
        "https://uploads.linear.app/a/good.png",
    ]
    assert out == [("/tmp/cached-image.png", "image/png")]
