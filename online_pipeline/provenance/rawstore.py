"""Content-addressed store for the original bytes of every ingested document.

An answer's figure can then be traced to the exact bytes it came from: the chunk
carries ``content_sha256``, and the store holds the file under that hash, per
tenant. Objects are written once and are never overwritten.
"""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
from pathlib import Path

# No dots: ".." must not survive as a path component.
_TENANT_SAFE = re.compile(r"[^A-Za-z0-9_-]")
_SHA = re.compile(r"[0-9a-f]{64}")


class RawStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def _path(self, tenant_id: str, sha256: str) -> Path:
        if not _SHA.fullmatch(sha256):
            raise ValueError("sha256 must be 64 lowercase hex characters")
        # Tenant ids come from a token but still become a directory name. A readable
        # prefix plus a hash suffix keeps distinct ids ("a/b" vs "a_b") apart and
        # can never be "..", absolute, or contain a separator.
        readable = _TENANT_SAFE.sub("_", tenant_id)[:40] or "_"
        suffix = hashlib.sha256(tenant_id.encode("utf-8", "surrogatepass")).hexdigest()[:12]
        return self.root / f"{readable}-{suffix}" / sha256[:2] / sha256

    def put_file(self, tenant_id: str, source: str | Path, sha256: str) -> Path:
        """Copy *source* into the store under *sha256* unless it is already there."""
        target = self._path(tenant_id, sha256)
        if target.exists():
            return target
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        # Write to a temp name and rename, so readers never see a partial object.
        fd, tmp = tempfile.mkstemp(dir=target.parent)
        try:
            digest = hashlib.sha256()
            with os.fdopen(fd, "wb") as out, open(source, "rb") as src:
                while block := src.read(1024 * 1024):
                    digest.update(block)
                    out.write(block)
            if digest.hexdigest() != sha256:
                raise ValueError("content does not match the declared sha256")
            os.chmod(tmp, 0o600)
            os.replace(tmp, target)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        return target

    def put_bytes(self, tenant_id: str, data: bytes, sha256: str | None = None) -> str:
        sha256 = sha256 or hashlib.sha256(data).hexdigest()
        fd, tmp = tempfile.mkstemp()
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
            self.put_file(tenant_id, tmp, sha256)
        finally:
            Path(tmp).unlink(missing_ok=True)
        return sha256

    def exists(self, tenant_id: str, sha256: str) -> bool:
        return self._path(tenant_id, sha256).exists()

    def get(self, tenant_id: str, sha256: str) -> bytes:
        return self._path(tenant_id, sha256).read_bytes()

    def delete(self, tenant_id: str, sha256: str) -> bool:
        path = self._path(tenant_id, sha256)
        if not path.exists():
            return False
        path.unlink()
        return True
