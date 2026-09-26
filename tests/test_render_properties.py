"""Card links from untrusted feeds: only http(s) may reach href/window.open."""

from __future__ import annotations

from hypothesis import given, strategies as st

from pipeline.render import _web_url

_SCHEMES = st.sampled_from(
    ["http", "https", "javascript", "data", "vbscript", "file", "ftp", "blob"]
)


def _browser_scheme(url: str) -> str | None:
    """Spec (WHATWG URL parsing, the parts that matter for a scheme): strip
    leading/trailing C0 controls and spaces, drop every tab/CR/LF, then the
    scheme is an ASCII letter followed by letters, digits, + - . up to ':'."""
    url = url.strip("".join(map(chr, range(0x21))))
    url = url.replace("\t", "").replace("\n", "").replace("\r", "")
    head, sep, _ = url.partition(":")
    if not sep or not head or not head[0].isascii() or not head[0].isalpha():
        return None
    if not all(c.isascii() and (c.isalnum() or c in "+-.") for c in head):
        return None
    return head.lower()


@st.composite
def _hostile_urls(draw: st.DrawFn) -> str:
    scheme = draw(_SCHEMES)
    # Mixed case and embedded tab/newline: browsers still read the scheme.
    scheme = "".join(
        draw(st.sampled_from([c.lower(), c.upper()]))
        + draw(st.sampled_from(["", "", "\t", "\n"]))
        for c in scheme
    )
    lead = draw(st.text(alphabet=" \t\n\x00\x01\x1f", max_size=3))
    rest = draw(st.sampled_from(["//example.com/a", "alert(1)", "text/html,<b>"]))
    return lead + scheme + ":" + rest


@given(st.one_of(_hostile_urls(), st.text(max_size=30)))
def test_card_links_are_http_or_empty(url: str) -> None:
    out = _web_url(url)
    assert out in ("", url)  # never rewritten, only dropped
    if out:
        assert _browser_scheme(out) in ("http", "https")


@given(_hostile_urls())
def test_real_web_links_survive(url: str) -> None:
    if _browser_scheme(url) in ("http", "https"):
        assert _web_url(url) == url
