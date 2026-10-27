"""XML parsing that cannot be turned against the server.

Rejects any document with a DTD or entity declaration (XXE, billion-laughs,
external-DTD fetches) and never touches the network. Used for DOCX parts today
and for (i)XBRL later.
"""

from __future__ import annotations

from typing import Any

from defusedxml import DefusedXmlException
from defusedxml import ElementTree as SafeET

from offline_pipeline.safety.errors import UnsafeContent

MAX_XML_BYTES = 50 * 1024 * 1024


def safe_fromstring(data: bytes | str) -> Any:
    """Parse *data* with DTDs, entities and external references forbidden."""
    size = len(data)
    if size > MAX_XML_BYTES:
        raise UnsafeContent("xml_too_large", "XML part is too large.")
    try:
        return SafeET.fromstring(data, forbid_dtd=True)
    except DefusedXmlException as exc:
        raise UnsafeContent("xml_unsafe", "XML uses DTD or entity features.") from exc
    except SafeET.ParseError as exc:
        raise UnsafeContent("xml_malformed", "XML could not be parsed.") from exc
    except RecursionError as exc:
        raise UnsafeContent("xml_too_deep", "XML is nested too deeply.") from exc
