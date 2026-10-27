"""Hand-built PDFs for the security tests (no PDF-writing dependency needed)."""

from __future__ import annotations

import io
import zlib


def build_pdf(
    content: bytes = b"BT /F1 12 Tf 72 700 Td (Hello visible world) Tj ET",
    *,
    catalog_extra: bytes = b"",
    page_extra: bytes = b"",
    extra_objects: list[bytes] | None = None,
    compress_content: bool = False,
    kids_self_reference: bool = False,
) -> bytes:
    """Return a minimal valid PDF. ``extra_objects`` become objects 6, 7, ..."""
    stream = zlib.compress(content) if compress_content else content
    stream_dict = b"/Filter /FlateDecode " if compress_content else b""
    kids = b"[2 0 R]" if kids_self_reference else b"[3 0 R]"
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R " + catalog_extra + b" >>",
        b"<< /Type /Pages /Kids " + kids + b" /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> " + page_extra + b" >>",
        b"<< " + stream_dict + b"/Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ] + list(extra_objects or [])

    out = io.BytesIO()
    out.write(b"%PDF-1.7\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(out.tell())
        out.write(f"{number} 0 obj\n".encode() + body + b"\nendobj\n")
    xref_at = out.tell()
    out.write(f"xref\n0 {len(objects) + 1}\n".encode())
    out.write(b"0000000000 65535 f \n")
    for offset in offsets:
        out.write(f"{offset:010d} 00000 n \n".encode())
    out.write(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_at}\n%%EOF\n".encode()
    )
    return out.getvalue()


def text_block(text: str, *, size: int = 12, x: int = 72, y: int = 700, color: str = "0 0 0") -> bytes:
    return f"{color} rg BT /F1 {size} Tf {x} {y} Td ({text}) Tj ET\n".encode()
