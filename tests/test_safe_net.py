"""SSRF-safe fetching: address classification, URL policy, redirects, limits."""

from __future__ import annotations

import gzip
import json
import time

import httpx
import pytest

from offline_pipeline.safety.errors import UnsafeContent
from offline_pipeline.safety.net import (
    SafeFetcher,
    is_public_ip,
    sniff,
    validate_url,
)

PUBLIC_IP = "93.184.216.34"
HTML = b"<html><head><title>t</title></head><body>hello</body></html>"


# ── address classification ───────────────────────────────────────────────────

class TestIsPublicIp:
    @pytest.mark.parametrize("ip", [
        "8.8.8.8", "93.184.216.34", "1.1.1.1", "2606:4700:4700::1111",
    ])
    def test_public_addresses_pass(self, ip):
        assert is_public_ip(ip)

    @pytest.mark.parametrize("ip", [
        "127.0.0.1", "127.255.255.254", "10.0.0.1", "172.16.5.4", "172.31.255.255",
        "192.168.1.1", "169.254.169.254",          # cloud metadata
        "100.64.0.1",                              # CGNAT / shared address space
        "0.0.0.0", "255.255.255.255", "224.0.0.1",  # unspecified / broadcast / multicast
        "192.0.2.1", "198.18.0.1",                 # documentation / benchmarking
        "::1", "::", "fe80::1", "fc00::1", "ff02::1",
        "::ffff:127.0.0.1", "::ffff:10.0.0.5", "::ffff:169.254.169.254",  # IPv4-mapped
        "2002:7f00:0001::1",                       # 6to4 wrapping 127.0.0.1
        "2001:0:4136:e378:8000:63bf:3fff:fdd2",    # Teredo
        "64:ff9b::7f00:1",                         # NAT64 wrapping loopback
        "not-an-ip", "",
    ])
    def test_non_public_addresses_are_blocked(self, ip):
        assert not is_public_ip(ip)


# ── URL policy ───────────────────────────────────────────────────────────────

ALLOW = ("sec.gov",)


class TestValidateUrl:
    def test_accepts_allowed_host_and_returns_parts(self):
        assert validate_url("https://www.SEC.gov./Archives/x?y=1#frag", ALLOW) == (
            "https", "www.sec.gov", 443, "/Archives/x?y=1",
        )

    @pytest.mark.parametrize("url,code", [
        ("http://sec.gov/x", "url_scheme"),
        ("ftp://sec.gov/x", "url_scheme"),
        ("file:///etc/passwd", "url_scheme"),
        ("javascript:alert(1)", "url_scheme"),
        ("gopher://sec.gov/", "url_scheme"),
        ("https://user:pw@sec.gov/x", "url_credentials"),
        ("https://sec.gov@evil.com/x", "url_credentials"),
        ("https://127.0.0.1/", "url_ip_literal"),
        ("https://[::1]/", "url_ip_literal"),
        ("https://169.254.169.254/latest/meta-data", "url_ip_literal"),
        ("https://sec.gov:8443/x", "url_port"),
        ("https://evil.com/x", "host_not_allowed"),
        ("https://sec.gov.evil.com/x", "host_not_allowed"),
        ("https://evilsec.gov/x", "host_not_allowed"),
        ("https://2130706433/", "host_not_allowed"),       # decimal-encoded 127.0.0.1
        ("https://0x7f000001/", "host_not_allowed"),         # hex-encoded
        ("https:///nohost", "url_invalid"),
        ("", "url_invalid"),
        ("https://sec.gov/" + "a" * 3000, "url_invalid"),
        ("https://sec.gov:99999/", "url_invalid"),
    ])
    def test_hostile_urls_are_refused(self, url, code):
        with pytest.raises(UnsafeContent) as exc:
            validate_url(url, ALLOW)
        assert exc.value.code == code

    def test_http_only_when_explicitly_allowed(self):
        with pytest.raises(UnsafeContent):
            validate_url("http://sec.gov/x", ALLOW)
        assert validate_url("http://sec.gov/x", ALLOW, allow_http=True, allowed_ports=(80, 443))[2] == 80

    def test_empty_allowlist_allows_nothing(self):
        with pytest.raises(UnsafeContent) as exc:
            validate_url("https://sec.gov/x", ())
        assert exc.value.code == "host_not_allowed"


# ── content sniffing ─────────────────────────────────────────────────────────

class TestSniff:
    @pytest.mark.parametrize("body,expected", [
        (b"%PDF-1.7\n...", "pdf"),
        (b"\xef\xbb\xbf  <!DOCTYPE html><p>x", "html"),
        (b"<html><body>x</body></html>", "html"),
        (b"<?xml version='1.0'?><a/>", "xml"),
        (b"<root><a/></root>", "xml"),
        (b'{"a": 1}', "json"),
        (b"[1, 2]", "json"),
        (b"{not json", "text"),
        (b"plain words", "text"),
        (b"abc\x00def", "binary"),
        (b"MZ\x90\x00\x03", "binary"),
    ])
    def test_classification(self, body, expected):
        assert sniff(body) == expected

    def test_pdf_header_hidden_after_junk_is_still_a_pdf(self):
        assert sniff(b"junk" * 50 + b"%PDF-1.4 rest") == "pdf"


# ── fetching ─────────────────────────────────────────────────────────────────

def make_fetcher(handler, *, resolver=None, allowlist=("sec.gov",), **kwargs):
    return SafeFetcher(
        allowlist=allowlist,
        resolver=resolver or (lambda host: [PUBLIC_IP]),
        transport=httpx.MockTransport(handler),
        **kwargs,
    )


def ok(content=HTML, content_type="text/html", **headers):
    return httpx.Response(200, content=content, headers={"content-type": content_type, **headers})


class TestFetch:
    def test_success_returns_hash_final_url_and_content(self):
        result = make_fetcher(lambda r: ok()).fetch("https://www.sec.gov/page")
        assert result.body == HTML and result.family == "html"
        assert result.final_url == "https://www.sec.gov/page"
        assert len(result.sha256) == 64 and result.fetched_at.endswith("+00:00")

    def test_connection_is_pinned_to_the_validated_ip_with_original_host(self):
        seen = {}

        def handler(request):
            seen["host"] = request.url.host
            seen["host_header"] = request.headers["host"]
            seen["sni"] = request.extensions.get("sni_hostname")
            seen["auth"] = request.headers.get("authorization")
            seen["cookie"] = request.headers.get("cookie")
            return ok()

        make_fetcher(handler).fetch("https://www.sec.gov/page")
        assert seen["host"] == PUBLIC_IP          # connected to the IP we validated
        assert seen["host_header"] == "www.sec.gov"
        assert seen["sni"] == "www.sec.gov"       # TLS still verifies the real name
        assert seen["auth"] is None and seen["cookie"] is None

    def test_dns_rebinding_second_answer_is_never_used(self):
        answers = iter([[PUBLIC_IP], ["127.0.0.1"]])
        calls = []

        def handler(request):
            calls.append(request.url.host)
            return ok()

        fetcher = make_fetcher(handler, resolver=lambda host: next(answers))
        fetcher.fetch("https://www.sec.gov/page")
        assert calls == [PUBLIC_IP]  # resolved once, connected once, to the checked address

    @pytest.mark.parametrize("answers", [
        ["10.0.0.5"], ["127.0.0.1"], ["169.254.169.254"], ["::1"],
        [PUBLIC_IP, "10.0.0.5"],            # one private record among public ones
        ["::ffff:127.0.0.1"],
    ])
    def test_hostnames_resolving_to_private_space_are_blocked(self, answers):
        called = []
        fetcher = make_fetcher(lambda r: called.append(r) or ok(), resolver=lambda h: answers)
        with pytest.raises(UnsafeContent) as exc:
            fetcher.fetch("https://www.sec.gov/page")
        assert exc.value.code == "ssrf_blocked" and not called

    def test_dns_failure_is_reported(self):
        def boom(host):
            raise OSError("nxdomain")

        with pytest.raises(UnsafeContent) as exc:
            make_fetcher(lambda r: ok(), resolver=boom).fetch("https://www.sec.gov/")
        assert exc.value.code == "dns_failure"
        with pytest.raises(UnsafeContent):
            make_fetcher(lambda r: ok(), resolver=lambda h: []).fetch("https://www.sec.gov/")


class TestRedirects:
    def test_allowed_redirect_is_followed_and_recorded(self):
        def handler(request):
            if request.headers["host"] == "sec.gov":
                return httpx.Response(301, headers={"location": "https://www.sec.gov/final"})
            return ok()

        result = make_fetcher(handler).fetch("https://sec.gov/start")
        assert result.final_url == "https://www.sec.gov/final"
        assert result.redirects == ["https://sec.gov/start"]

    def test_relative_redirect_is_resolved(self):
        hops = []

        def handler(request):
            hops.append(request.url.path)
            if request.url.path == "/a":
                return httpx.Response(302, headers={"location": "/b"})
            return ok()

        assert make_fetcher(handler).fetch("https://sec.gov/a").final_url == "https://sec.gov/b"

    @pytest.mark.parametrize("target,code", [
        ("https://evil.com/x", "host_not_allowed"),
        ("http://sec.gov/x", "url_scheme"),
        ("https://127.0.0.1/x", "url_ip_literal"),
        ("https://user@sec.gov/x", "url_credentials"),
        ("file:///etc/passwd", "url_scheme"),
    ])
    def test_redirect_targets_are_revalidated(self, target, code):
        fetcher = make_fetcher(lambda r: httpx.Response(302, headers={"location": target}))
        with pytest.raises(UnsafeContent) as exc:
            fetcher.fetch("https://sec.gov/start")
        assert exc.value.code == code

    def test_redirect_to_an_allowed_name_that_resolves_privately_is_blocked(self):
        def resolver(host):
            return ["10.1.1.1"] if host == "internal.sec.gov" else [PUBLIC_IP]

        def handler(request):
            return httpx.Response(302, headers={"location": "https://internal.sec.gov/admin"})

        with pytest.raises(UnsafeContent) as exc:
            make_fetcher(handler, resolver=resolver).fetch("https://sec.gov/start")
        assert exc.value.code == "ssrf_blocked"

    def test_redirect_chains_are_capped(self):
        count = []

        def handler(request):
            count.append(1)
            return httpx.Response(302, headers={"location": "https://sec.gov/next"})

        with pytest.raises(UnsafeContent) as exc:
            make_fetcher(handler, max_redirects=3).fetch("https://sec.gov/start")
        assert exc.value.code == "too_many_redirects" and len(count) == 4

    def test_redirect_without_location_is_rejected(self):
        with pytest.raises(UnsafeContent) as exc:
            make_fetcher(lambda r: httpx.Response(302)).fetch("https://sec.gov/x")
        assert exc.value.code == "redirect_invalid"


class TestLimits:
    def test_declared_content_length_over_the_cap_is_refused_early(self):
        fetcher = make_fetcher(lambda r: ok(b"x" * 10, **{"content-length": "999999999"}), max_bytes=1000)
        with pytest.raises(UnsafeContent) as exc:
            fetcher.fetch("https://sec.gov/x")
        assert exc.value.code == "response_too_large"

    def test_streamed_body_over_the_cap_is_cut_off(self):
        def chunks():
            for _ in range(100):
                yield b"x" * 100

        def handler(request):
            return httpx.Response(200, content=chunks(), headers={"content-type": "text/plain"})

        with pytest.raises(UnsafeContent) as exc:
            make_fetcher(handler, max_bytes=1000).fetch("https://sec.gov/x")
        assert exc.value.code == "response_too_large"

    def test_gzip_bomb_is_capped_on_decoded_size(self):
        bomb = gzip.compress(b"A" * 5_000_000)  # a few KB on the wire
        assert len(bomb) < 20_000

        def handler(request):
            return httpx.Response(200, content=bomb, headers={"content-type": "text/plain", "content-encoding": "gzip"})

        with pytest.raises(UnsafeContent) as exc:
            make_fetcher(handler, max_bytes=100_000).fetch("https://sec.gov/x")
        assert exc.value.code == "response_too_large"

    def test_slow_download_hits_the_total_deadline(self):
        now = [0.0]

        def clock():
            now[0] += 10.0
            return now[0]

        def handler(request):
            return httpx.Response(200, content=iter([b"x" * 10] * 50), headers={"content-type": "text/plain"})

        fetcher = make_fetcher(handler, total_timeout=25.0, clock=clock)
        with pytest.raises(UnsafeContent) as exc:
            fetcher.fetch("https://sec.gov/x")
        assert exc.value.code == "fetch_timeout"

    def test_transport_timeout_and_errors_are_mapped(self):
        def slow(request):
            raise httpx.ReadTimeout("slow", request=request)

        def broken(request):
            raise httpx.ConnectError("refused", request=request)

        with pytest.raises(UnsafeContent) as exc:
            make_fetcher(slow).fetch("https://sec.gov/x")
        assert exc.value.code == "fetch_timeout"
        with pytest.raises(UnsafeContent) as exc:
            make_fetcher(broken).fetch("https://sec.gov/x")
        assert exc.value.code == "fetch_failed"

    @pytest.mark.parametrize("status", [400, 403, 404, 500, 503])
    def test_http_errors_are_rejected(self, status):
        with pytest.raises(UnsafeContent) as exc:
            make_fetcher(lambda r: httpx.Response(status, content=b"no")).fetch("https://sec.gov/x")
        assert exc.value.code == "http_error"


class TestContentTypeChecks:
    @pytest.mark.parametrize("declared", [
        "application/octet-stream", "application/x-msdownload", "image/png",
        "application/zip", "application/javascript", "text/javascript", "",
    ])
    def test_disallowed_declared_types_are_blocked(self, declared):
        with pytest.raises(UnsafeContent) as exc:
            make_fetcher(lambda r: ok(b"data", declared)).fetch("https://sec.gov/x")
        assert exc.value.code == "content_type_blocked"

    @pytest.mark.parametrize("body,declared", [
        (b"%PDF-1.7 exploit", "text/html"),       # PDF pretending to be a page
        (HTML, "application/pdf"),                 # page pretending to be a PDF
        (b"MZ\x90\x00\x00binary", "text/plain"),   # executable as text
        (b"<html>x</html>", "application/json"),
        (b"not json at all", "application/json"),
    ])
    def test_declared_type_must_match_the_bytes(self, body, declared):
        with pytest.raises(UnsafeContent) as exc:
            make_fetcher(lambda r: ok(body, declared)).fetch("https://sec.gov/x")
        assert exc.value.code == "type_mismatch"

    def test_charset_parameters_are_ignored_for_matching(self):
        result = make_fetcher(lambda r: ok(HTML, "text/html; charset=UTF-8")).fetch("https://sec.gov/x")
        assert result.family == "html"

    def test_json_is_accepted_for_api_fetchers_only(self):
        body = json.dumps({"a": 1}).encode()
        api = make_fetcher(lambda r: ok(body, "application/json"),
                           allowed_content_types={"application/json": "json"})
        assert api.fetch("https://sec.gov/api").family == "json"
        pages = make_fetcher(lambda r: ok(body, "application/json"),
                             allowed_content_types={"text/html": "html"})
        with pytest.raises(UnsafeContent) as exc:
            pages.fetch("https://sec.gov/api")
        assert exc.value.code == "content_type_blocked"
