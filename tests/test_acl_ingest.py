"""P2: ACL construction, tenant/owner-scoped ids, and ACL stamping on ingest."""

from __future__ import annotations

import json
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import jwt
import pytest
from fastapi.testclient import TestClient

from config.settings import Settings
from online_pipeline.auth.acl import (
    ACLError,
    build_document_acl,
    document_id,
    has_acl,
)
from online_pipeline.auth.policy import TenantPolicy, TenantPolicyStore
from online_pipeline.auth.principal import Principal
from online_pipeline.auth.tokens import TokenValidator

DEV_SECRET = "d" * 40


def principal(**kw) -> Principal:
    base = dict(user_id="alice", tenant_id="eq", groups=frozenset({"fx", "rates"}),
                clearance=2, scopes=frozenset({"rag:ingest"}))
    base.update(kw)
    return Principal(**base)


# ── build_document_acl ───────────────────────────────────────────────────────

class TestBuildDocumentAcl:
    policy = TenantPolicy(max_classification=2)

    def test_default_is_private_to_the_owner(self):
        acl = build_document_acl(principal(), self.policy)
        assert acl.as_payload() == {
            "tenant_id": "eq", "owner_id": "alice", "acl_groups": [],
            "acl_public": False, "classification": 1,
        }

    def test_tenant_and_owner_always_come_from_the_principal(self):
        acl = build_document_acl(principal(user_id="bob", tenant_id="ib"), self.policy)
        assert (acl.tenant_id, acl.owner_id) == ("ib", "bob")

    def test_can_share_with_own_groups_only(self):
        acl = build_document_acl(principal(), self.policy, acl_groups=["fx", " fx ", "rates", ""])
        assert acl.acl_groups == ("fx", "rates")
        with pytest.raises(ACLError) as exc:
            build_document_acl(principal(), self.policy, acl_groups=["fx", "desk-secret"])
        assert exc.value.code == "group_not_held"

    @pytest.mark.parametrize("level,code", [
        (3, "above_clearance"),        # principal clearance is 2
        (-1, "bad_classification"),
        (4, "bad_classification"),
        (True, "bad_classification"),
    ])
    def test_classification_limits(self, level, code):
        with pytest.raises(ACLError) as exc:
            build_document_acl(principal(), self.policy, classification=level)
        assert exc.value.code == code

    def test_tenant_ceiling_caps_classification(self):
        with pytest.raises(ACLError) as exc:
            build_document_acl(principal(clearance=3), TenantPolicy(max_classification=1), classification=2)
        assert exc.value.code == "above_tenant_max"

    def test_default_classification_never_exceeds_clearance_or_ceiling(self):
        assert build_document_acl(principal(clearance=0), self.policy).classification == 0
        assert build_document_acl(principal(clearance=3), TenantPolicy(max_classification=0)).classification == 0

    def test_group_limits(self):
        with pytest.raises(ACLError):
            build_document_acl(principal(groups=frozenset(f"g{i}" for i in range(80))),
                               self.policy, acl_groups=[f"g{i}" for i in range(60)])
        with pytest.raises(ACLError):
            build_document_acl(principal(groups=frozenset({"x" * 200})), self.policy, acl_groups=["x" * 200])


# ── document ids ─────────────────────────────────────────────────────────────

class TestDocumentId:
    def test_deterministic(self):
        assert document_id("t", "u", "text") == document_id("t", "u", "text")

    def test_identical_content_differs_by_tenant_and_by_owner(self):
        base = document_id("t1", "u1", "same bytes")
        assert document_id("t2", "u1", "same bytes") != base
        assert document_id("t1", "u2", "same bytes") != base

    def test_field_boundaries_cannot_be_shifted(self):
        assert document_id("ab", "c", "x") != document_id("a", "bc", "x")


class TestHasAcl:
    def test_requires_every_field(self):
        full = {"tenant_id": "t", "owner_id": "u", "acl_groups": [], "acl_public": False, "classification": 1}
        assert has_acl(full)
        for field in full:
            assert not has_acl({k: v for k, v in full.items() if k != field})


# ── Ingest endpoints ─────────────────────────────────────────────────────────

def make_api(audit_log):
    from online_pipeline.api import gateway
    from online_pipeline.auth import deps

    cfg = Settings(_env_file=None, app_env="dev", auth_mode="dev_token",
                   dev_jwt_secret=DEV_SECRET, audit_hmac_key="k")
    store = TenantPolicyStore(tenants={
        "eq": TenantPolicy(max_classification=2, allow_url_ingest=True, url_allowlist=("sec.gov",)),
        "ib": TenantPolicy(max_classification=3),
    })
    upsert = MagicMock()
    vector_db = SimpleNamespace(upsert=upsert)
    patches = [
        patch.object(deps, "get_validator", return_value=TokenValidator(cfg)),
        patch.object(gateway, "_policies", store),
        patch.object(gateway, "_vector_db", vector_db),
        patch.object(gateway, "embed_texts", side_effect=lambda texts: [[0.0, 0.0] for _ in texts]),
    ]
    for p in patches:
        p.start()
    return SimpleNamespace(client=TestClient(gateway.app), gateway=gateway, upsert=upsert,
                           audit=audit_log, stop=lambda: [p.stop() for p in reversed(patches)])


@pytest.fixture
def api(audit_log):
    handle = make_api(audit_log)
    yield handle
    handle.stop()


def bearer(user="alice", tenant="eq", groups=("fx", "rates"), clearance=2,
           scopes=("rag:ingest",)) -> dict:
    now = int(time.time())
    token = jwt.encode({
        "iss": "dev-issuer", "aud": "api://default", "sub": user, "iat": now, "exp": now + 600,
        "tenant_id": tenant, "groups": list(groups), "clearance": clearance, "scp": list(scopes),
    }, DEV_SECRET, algorithm="HS256")
    return {"Authorization": f"Bearer {token}"}


def upload(api, content=b"hello world " * 40, headers=None, data=None, name="a.txt"):
    with patch.object(api.gateway, "DocumentExtractor") as extractor:
        extractor.return_value.extract.return_value = {
            "text": content.decode(), "metadata": {"source": name, "file_type": "txt"},
        }
        return api.client.post(
            "/ingest", headers=headers or bearer(), data=data or {},
            files=[("files", (name, content, "text/plain"))],
        )


def upserted_points(api):
    return [p for call in api.upsert.call_args_list for p in call.args[0]]


class TestIngestStampsAcl:
    def test_every_chunk_carries_the_acl_from_the_token(self, api):
        response = upload(api, data={"acl_groups": ["fx"], "classification": "2"})
        assert response.status_code == 200
        points = upserted_points(api)
        assert points
        for point in points:
            assert point["payload"]["tenant_id"] == "eq"
            assert point["payload"]["owner_id"] == "alice"
            assert point["payload"]["acl_groups"] == ["fx"]
            assert point["payload"]["acl_public"] is False
            assert point["payload"]["classification"] == 2

    def test_default_upload_is_private_and_internal(self, api):
        upload(api)
        payload = upserted_points(api)[0]["payload"]
        assert (payload["acl_groups"], payload["acl_public"], payload["classification"]) == ([], False, 1)

    def test_client_cannot_choose_tenant_or_owner(self, api):
        upload(api, data={"tenant_id": "ib", "owner_id": "mallory", "tenant": "ib"})
        for point in upserted_points(api):
            assert point["payload"]["tenant_id"] == "eq"
            assert point["payload"]["owner_id"] == "alice"

    def test_extractor_supplied_acl_fields_cannot_override_the_stamp(self, api):
        with patch.object(api.gateway, "DocumentExtractor") as extractor:
            extractor.return_value.extract.return_value = {
                "text": "payload " * 50,
                "metadata": {"tenant_id": "ib", "acl_public": True, "classification": 0, "owner_id": "x"},
            }
            api.client.post("/ingest", headers=bearer(),
                            files=[("files", ("a.txt", b"x", "text/plain"))])
        for point in upserted_points(api):
            assert point["payload"]["tenant_id"] == "eq" and point["payload"]["acl_public"] is False

    @pytest.mark.parametrize("data,reason_code", [
        ({"acl_groups": ["desk-secret"]}, "group_not_held"),
        ({"classification": "3"}, "above_clearance"),
        ({"classification": "9"}, "bad_classification"),
    ])
    def test_forbidden_acl_rejects_the_request_before_anything_is_indexed(self, api, data, reason_code):
        response = upload(api, data=data)
        assert response.status_code == 403
        api.upsert.assert_not_called()
        event = [json.loads(l) for l in api.audit.path.read_text().splitlines()][-1]
        assert (event["action"], event["reason"]) == ("policy.deny", "acl_not_permitted")
        assert event["details"]["code"] == reason_code

    def test_tenant_ceiling_applies_to_classification(self, api):
        response = upload(api, headers=bearer(clearance=3), data={"classification": "3"})
        assert response.status_code == 403  # tenant "eq" caps at 2

    def test_url_ingest_stamps_acl_too(self, api):
        with patch.object(api.gateway, "WebExtractor") as extractor:
            extractor.return_value.extract.return_value = {"text": "page " * 80, "metadata": {"source": "u"}}
            response = api.client.post(
                "/ingest/url", headers=bearer(),
                json={"url": "https://www.sec.gov/x", "acl_groups": ["rates"], "classification": 1},
            )
        assert response.status_code == 200
        payload = upserted_points(api)[0]["payload"]
        assert payload["acl_groups"] == ["rates"] and payload["tenant_id"] == "eq"

    def test_url_ingest_rejects_group_not_held(self, api):
        with patch.object(api.gateway, "WebExtractor") as extractor:
            response = api.client.post(
                "/ingest/url", headers=bearer(),
                json={"url": "https://www.sec.gov/x", "acl_groups": ["not-mine"]},
            )
        assert response.status_code == 403
        extractor.assert_not_called()


class TestIdentityOfIdenticalUploads:
    def ids(self, api):
        return {p["id"] for p in upserted_points(api)}

    def test_two_tenants_uploading_identical_bytes_do_not_collide(self, api):
        upload(api, headers=bearer(tenant="eq"))
        a = self.ids(api)
        api.upsert.reset_mock()
        upload(api, headers=bearer(tenant="ib"))
        b = self.ids(api)
        assert a and b and a.isdisjoint(b)

    def test_two_users_in_one_tenant_cannot_overwrite_each_others_acl(self, api):
        upload(api, headers=bearer(user="alice"), data={"acl_groups": ["fx"]})
        alice = {p["id"]: p["payload"] for p in upserted_points(api)}
        api.upsert.reset_mock()
        upload(api, headers=bearer(user="bob"), data={"acl_public": "true"})
        bob = {p["id"]: p["payload"] for p in upserted_points(api)}
        assert set(alice).isdisjoint(bob)
        assert all(p["owner_id"] == "alice" and not p["acl_public"] for p in alice.values())

    def test_same_owner_reingest_is_idempotent_and_updates_the_acl(self, api):
        upload(api, data={"acl_groups": ["fx"]})
        first = {p["id"]: p["payload"]["acl_groups"] for p in upserted_points(api)}
        api.upsert.reset_mock()
        upload(api, data={"acl_groups": ["rates"]})
        second = {p["id"]: p["payload"]["acl_groups"] for p in upserted_points(api)}
        assert set(first) == set(second)
        assert all(g == ["rates"] for g in second.values())


class TestCollectionSeamAndIndexes:
    def test_payload_indexes_are_created_for_every_filtered_field(self):
        from vector_db.client import VectorDBClient

        client = VectorDBClient(url=":memory:", collection="c", dimension=2)
        client._client = MagicMock()
        client._client.get_collections.return_value.collections = []
        client.ensure_collection()
        fields = {c.kwargs["field_name"] for c in client._client.create_payload_index.call_args_list}
        assert {"tenant_id", "owner_id", "acl_groups", "classification", "acl_public", "doc_id"} <= fields

    def test_collection_seam_returns_the_shared_collection_today(self):
        from vector_db.client import VectorDBClient

        client = VectorDBClient(url=":memory:", collection="shared", dimension=2)
        assert client.collection_for_tenant("any-tenant") == "shared"
