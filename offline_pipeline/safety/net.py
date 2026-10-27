"""SSRF-safe, size-capped HTTP fetching for every outbound ingest request.

All of ``/ingest/url``, ``/ingest/api`` (and later XBRL taxonomy fetches) go
through :class:`SafeFetcher`:

* https only, no credentials in the URL, fixed ports, host on the tenant allow-list
* every resolved address must be globally routable (no loopback, RFC1918,
  link-local/cloud-metadata, CGNAT, multicast, IPv4-mapped/6to4/NAT64 tricks)
* the connection is made to the *validated* IP (defeats DNS rebinding) while TLS
  still verifies the original host name
* redirects are followed manually, each hop re-validated, at most a few
* the body is streamed with a hard cap on decoded bytes and a wall-clock deadline
* the declared Content-Type must be allow-listed AND match what the bytes are
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import socket
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Sequence
from urllib.parse import urljoin, urlsplit

import httpx

from offline_pipeline.safety.errors import UnsafeContent

MAX_URL_LENGTH = 2048

ALLOWED_CONTENT_TYPES: dict[str, str] = {
    "text/html": "html",
    "application/xhtml+xml": "html",
    "application/json": "json",
    "application/xml": "xml",
    "text/xml": "xml",
    "application/pdf": "pdf",
    "text/plain": "text",
}

_NAT64 = ipaddress.ip_network("64:ff9b::/96")


def is_public_ip(value: str) -> bool:
    """True only for addresses that are safe to connect to from the server."""
    try:
        ip = ipaddress.ip_address(value)
    except ValueError:
        return False
    if isinstance(ip, ipaddress.IPv6Address):
        if ip.ipv4_mapped is not None:
            return is_public_ip(str(ip.ipv4_mapped))
        if ip.sixtofour is not None and not is_public_ip(str(ip.sixtofour)):
            return False
        if ip.teredo is not None:
            return False
        if ip in _NAT64:
            return False
    return bool(ip.is_global) and not ip.is_multicast


def validate_url(
    url: str,
    allowlist: Sequence[str],
    *,
    allow_http: bool = False,
    allowed_ports: Sequence[int] = (443,),
) -> tuple[str, str, int, str]:
    """Return (scheme, host, port, path+query) or raise :class:`UnsafeContent`."""
    url = (url or "").strip()
    if not url or len(url) > MAX_URL_LENGTH:
        raise UnsafeContent("url_invalid", "URL is empty or too long.")
    try:
        parts = urlsplit(url)
        port = parts.port
        raw_host = parts.hostname
    except ValueError as exc:
        raise UnsafeContent("url_invalid", "URL could not be parsed.") from exc

    schemes = {"https", "http"} if allow_http else {"https"}
    if parts.scheme not in schemes:
        raise UnsafeContent("url_scheme", "Only https URLs are allowed.")
    if parts.username is not None or parts.password is not None:
        raise UnsafeContent("url_credentials", "Credentials in URLs are not allowed.")
    if not raw_host:
        raise UnsafeContent("url_invalid", "URL has no host.")

    try:
        host = raw_host.encode("idna").decode("ascii").lower().rstrip(".")
    except UnicodeError as exc:
        raise UnsafeContent("url_invalid", "Host name is not valid.") from exc
    try:
        ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        raise UnsafeContent("url_ip_literal", "IP addresses are not allowed; use a host name.")

    default_port = 443 if parts.scheme == "https" else 80
    port = port or default_port
    if port not in allowed_ports and not (allow_http and port == default_port):
        raise UnsafeContent("url_port", "Port is not allowed.")

    if not any(host == d or host.endswith("." + d) for d in allowlist):
        raise UnsafeContent("host_not_allowed", "Host is not on the allow-list.")

    path = parts.path or "/"
    if parts.query:
        path += "?" + parts.query
    return parts.scheme, host, port, path


def default_resolver(host: str) -> list[str]:
    infos = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    return list(dict.fromkeys(info[4][0] for info in infos))


def sniff(body: bytes) -> str:
    """Classify *body* from its bytes alone: html, xml, pdf, json, text or binary."""
    head = body[:2048].lstrip(b"\xef\xbb\xbf \t\r\n")
    if body[:1024].find(b"%PDF-") != -1:
        return "pdf"
    if b"\x00" in body[:8192]:
        return "binary"
    lowered = head.lower()
    if lowered.startswith(b"<?xml"):
        return "xml"
    if lowered.startswith(b"<!doctype html") or b"<html" in lowered[:2048]:
        return "html"
    if lowered.startswith(b"<"):
        return "xml"
    if head[:1] in (b"{", b"["):
        try:
            json.loads(body)
            return "json"
        except ValueError:
            return "text"
    return "text"


# Declared family → sniffed families that are acceptable for it.
_COMPATIBLE = {
    "html": {"html", "xml"},   # XHTML
    "xml": {"xml", "html"},
    "json": {"json"},
    "pdf": {"pdf"},
    "text": {"text"},
}


@dataclass(frozen=True)
class FetchResult:
    url: str
    final_url: str
    content_type: str
    family: str
    body: bytes
    sha256: str
    fetched_at: str
    redirects: list[str] = field(default_factory=list)


class SafeFetcher:
    def __init__(
        self,
        *,
        allowlist: Sequence[str],
        max_bytes: int = 10 * 1024 * 1024,
        max_redirects: int = 3,
        connect_timeout: float = 5.0,
        read_timeout: float = 15.0,
        total_timeout: float = 30.0,
        allow_http: bool = False,
        allowed_ports: Sequence[int] = (443,),
        allowed_content_types: dict[str, str] | None = None,
        resolver: Callable[[str], list[str]] | None = None,
        transport: httpx.BaseTransport | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.allowlist = tuple(allowlist)
        self.max_bytes = max_bytes
        self.max_redirects = max_redirects
        self.timeout = httpx.Timeout(
            connect=connect_timeout, read=read_timeout, write=read_timeout, pool=connect_timeout
        )
        self.total_timeout = total_timeout
        self.allow_http = allow_http
        self.allowed_ports = tuple(allowed_ports)
        self.allowed_types = allowed_content_types or ALLOWED_CONTENT_TYPES
        self._resolve = resolver or default_resolver
        self._transport = transport
        self._clock = clock

    def fetch(self, url: str, params: dict[str, Any] | None = None) -> FetchResult:
        deadline = self._clock() + self.total_timeout
        current = url.strip()
        hops: list[str] = []
        with httpx.Client(
            transport=self._transport,
            timeout=self.timeout,
            follow_redirects=False,
            trust_env=False,  # proxy env vars would bypass the pinned address
        ) as client:
            for _ in range(self.max_redirects + 1):
                scheme, host, port, path = validate_url(
                    current, self.allowlist, allow_http=self.allow_http, allowed_ports=self.allowed_ports
                )
                ip = self._pick_address(host)
                response_target = self._pinned_url(scheme, ip, port, path)
                host_header = host if port in (80, 443) else f"{host}:{port}"
                extensions = {"sni_hostname": host} if scheme == "https" else {}

                try:
                    with client.stream(
                        "GET",
                        response_target,
                        params=params if not hops else None,
                        headers={"Host": host_header, "Accept": ", ".join(self.allowed_types)},
                        extensions=extensions,
                    ) as response:
                        if response.is_redirect:
                            location = response.headers.get("location", "")
                            if not location:
                                raise UnsafeContent("redirect_invalid", "Redirect without a target.")
                            hops.append(current)
                            current = urljoin(current, location)
                            continue
                        return self._consume(url, current, response, hops, deadline)
                except httpx.TimeoutException as exc:
                    raise UnsafeContent("fetch_timeout", "The remote server timed out.") from exc
                except httpx.HTTPError as exc:
                    raise UnsafeContent("fetch_failed", "The remote request failed.") from exc
        raise UnsafeContent("too_many_redirects", "Too many redirects.")

    # ── internals ────────────────────────────────────────────────────────────

    def _pick_address(self, host: str) -> str:
        try:
            addresses = self._resolve(host)
        except OSError as exc:
            raise UnsafeContent("dns_failure", "Host name did not resolve.") from exc
        if not addresses:
            raise UnsafeContent("dns_failure", "Host name did not resolve.")
        # Every answer must be public: one private record is enough to refuse,
        # otherwise a resolver could hand the private one out on a later lookup.
        if not all(is_public_ip(a) for a in addresses):
            raise UnsafeContent("ssrf_blocked", "Host resolves to a non-public address.")
        return addresses[0]

    @staticmethod
    def _pinned_url(scheme: str, ip: str, port: int, path: str) -> str:
        literal = f"[{ip}]" if ":" in ip else ip
        default = 443 if scheme == "https" else 80
        netloc = literal if port == default else f"{literal}:{port}"
        return f"{scheme}://{netloc}{path}"

    def _consume(
        self,
        original: str,
        final_url: str,
        response: httpx.Response,
        hops: list[str],
        deadline: float,
    ) -> FetchResult:
        if response.status_code >= 400:
            raise UnsafeContent("http_error", f"Remote server returned {response.status_code}.")

        declared = response.headers.get("content-type", "").split(";")[0].strip().lower()
        family = self.allowed_types.get(declared)
        if family is None:
            raise UnsafeContent("content_type_blocked", "Content type is not allowed.")

        length = response.headers.get("content-length")
        if length and length.isdigit() and int(length) > self.max_bytes:
            raise UnsafeContent("response_too_large", "Response is too large.")

        body = bytearray()
        # iter_bytes decodes gzip, so the cap bounds the *decompressed* size.
        for chunk in response.iter_bytes():
            body.extend(chunk)
            if len(body) > self.max_bytes:
                raise UnsafeContent("response_too_large", "Response is too large.")
            if self._clock() > deadline:
                raise UnsafeContent("fetch_timeout", "The download took too long.")

        data = bytes(body)
        if sniff(data) not in _COMPATIBLE[family]:
            raise UnsafeContent("type_mismatch", "Content does not match its declared type.")
        return FetchResult(
            url=original,
            final_url=final_url,
            content_type=declared,
            family=family,
            body=data,
            sha256=hashlib.sha256(data).hexdigest(),
            fetched_at=datetime.now(tz=timezone.utc).isoformat(),
            redirects=list(hops),
        )
