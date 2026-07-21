"""A malformed request is the CALLER's error (400), not a server crash (500).

Red-team finding: every POST failure — including a truncated body, a non-numeric Content-Length, or a JSON
array where an object was required — returned 500. That tells an integrator OUR server fell over, and makes
their own bug undebuggable. Genuine server-side failures must still be 500, so the two are separated.
"""
from __future__ import annotations

import json
from http import HTTPStatus


class _FakeRfile:
    def __init__(self, raw: bytes):
        self._raw = raw

    def read(self, n: int) -> bytes:
        return self._raw[:n]


class _Probe:
    """Exercises the real _handle_json_post against a fake request, capturing what it would send."""

    def __init__(self, body: str, content_length: str | None = None):
        from virturoid.ui_server import _Handler  # noqa: PLC0415
        raw = body.encode("utf-8")
        self.headers = {"Content-Length": content_length if content_length is not None else str(len(raw))}
        self.rfile = _FakeRfile(raw)
        self.sent: tuple = ()
        self._handle_json_post = _Handler._handle_json_post.__get__(self, _Probe)

    def _send_json(self, obj, status=HTTPStatus.OK):
        self.sent = (obj, status)


def _post(body: str, handler=lambda payload: {"ok": True, "echo": payload}, content_length=None):
    p = _Probe(body, content_length)
    p._handle_json_post(handler)
    return p.sent


def test_malformed_json_is_400_not_500():
    obj, status = _post('{"tool": "build",')                    # truncated body
    assert status == HTTPStatus.BAD_REQUEST, (obj, status)
    assert "bad request" in json.dumps(obj).lower()


def test_a_json_array_where_an_object_is_required_is_400():
    obj, status = _post("[1, 2, 3]")
    assert status == HTTPStatus.BAD_REQUEST
    assert "object" in json.dumps(obj).lower()


def test_a_bare_scalar_body_is_400():
    _, status = _post('"just a string"')
    assert status == HTTPStatus.BAD_REQUEST


def test_a_non_numeric_content_length_is_400():
    _, status = _post("{}", content_length="not-a-number")
    assert status == HTTPStatus.BAD_REQUEST


def test_an_empty_body_is_treated_as_an_empty_object():
    """A bodyless POST is valid — it means 'no arguments', not a malformed request."""
    obj, status = _post("", content_length="0")
    assert status == HTTPStatus.OK
    assert obj == {"ok": True, "echo": {}}


def test_a_valid_body_still_reaches_the_handler():
    obj, status = _post('{"tool": "build", "args": {"n": 1}}')
    assert status == HTTPStatus.OK
    assert obj["echo"]["tool"] == "build"


def test_a_genuine_handler_failure_is_still_500():
    """The 400 path must not swallow real server-side errors — that would hide OUR bugs."""
    def boom(_payload):
        raise RuntimeError("database is on fire")

    obj, status = _post("{}", handler=boom)
    assert status == HTTPStatus.INTERNAL_SERVER_ERROR
    assert "fire" in json.dumps(obj)
