"""Authentication, scopes, tenant policy and endpoint authorization."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient

from config.settings import Settings
from online_pipeline.auth.policy import (
    PolicyError,
    RESTRICTIVE,
    TenantPolicy,
    TenantPolicyStore,
    url_host_allowed,
    web_search_decision,
)
from online_pipeline.auth.principal import Principal
from online_pipeline.auth.tokens import AuthError, TokenValidator, validate_auth_config

ISSUER = "https://acme.okta.com/oauth2/default"
AUDIENCE = "api://default"
DEV_SECRET = "d" * 40


def make_settings(**overrides) -> Settings:
    base = dict(
        _env_file=None, app_env="dev", auth_mode="okta", okta_issuer=ISSUER,
        okta_audience=AUDIENCE, dev_jwt_secret=DEV_SECRET, audit_hmac_key="k",
    )
    base.update(overrides)
    return Settings(**base)


@pytest.fixture(scope="module")
def keypair():
    private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return SimpleNamespace(
        private_pem=private.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        ),
        public=private.public_key(),
        public_pem=private.public_key().public_bytes(
            serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
        ),
    )


@pytest.fixture(scope="module")
def other_keypair():
    private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return SimpleNamespace(private_pem=private.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))


def claims(**overrides) -> dict:
    now = int(time.time())
    base = {
        "iss": ISSUER, "aud": AUDIENCE, "sub": "alice@acme.com", "uid": "00u123",
        "iat": now, "exp": now + 600,
        "tenant_id": "equities", "groups": ["desk-fx", "research"],
        "clearance": 2, "scp": ["rag:query", "rag:ingest"],
    }
    base.update(overrides)
    return {k: v for k, v in base.items() if v is not None}


def okta_validator(keypair, **cfg) -> TokenValidator:
    validator = TokenValidator(make_settings(**cfg))
    validator._jwks = SimpleNamespace(
        get_signing_key_from_jwt=lambda token: SimpleNamespace(key=keypair.public)
    )
    return validator


def rs256(keypair, **overrides) -> str:
    return jwt.encode(claims(**overrides), keypair.private_pem, algorithm="RS256")


# ── Token validation (Okta mode) ─────────────────────────────────────────────

class TestOktaTokens:
    def test_valid_token_maps_claims_to_principal(self, keypair):
        principal = okta_validator(keypair).validate(rs256(keypair))
        assert principal == Principal(
            user_id="00u123", tenant_id="equities",
            groups=frozenset({"desk-fx", "research"}), clearance=2,
            scopes=frozenset({"rag:query", "rag:ingest"}),
        )

    def test_user_id_falls_back_to_sub(self, keypair):
        principal = okta_validator(keypair).validate(rs256(keypair, uid=None))
        assert principal.user_id == "alice@acme.com"

    @pytest.mark.parametrize("overrides,code", [
        ({"exp": int(time.time()) - 3600}, "token_expired"),
        ({"aud": "api://other"}, "bad_audience"),
        ({"iss": "https://evil.okta.com/oauth2/default"}, "bad_issuer"),
        ({"exp": None}, "missing_claim"),
        ({"sub": None}, "missing_claim"),
    ])
    def test_invalid_claims_are_rejected(self, keypair, overrides, code):
        with pytest.raises(AuthError) as exc:
            okta_validator(keypair).validate(rs256(keypair, **overrides))
        assert (exc.value.status_code, exc.value.code) == (401, code)

    def test_token_signed_by_another_key_is_rejected(self, keypair, other_keypair):
        token = jwt.encode(claims(), other_keypair.private_pem, algorithm="RS256")
        with pytest.raises(AuthError) as exc:
            okta_validator(keypair).validate(token)
        assert exc.value.code == "bad_signature"

    def test_alg_none_is_rejected(self, keypair):
        token = jwt.encode(claims(), key=None, algorithm="none")
        with pytest.raises(AuthError) as exc:
            okta_validator(keypair).validate(token)
        assert exc.value.status_code == 401

    def test_rs256_to_hs256_key_confusion_is_rejected(self, keypair):
        # Classic attack: sign HS256 using the public key as the HMAC secret.
        def b64(data: bytes) -> bytes:
            return base64.urlsafe_b64encode(data).rstrip(b"=")

        header = b64(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
        body = b64(json.dumps(claims()).encode())
        signing_input = header + b"." + body
        sig = b64(hmac.new(keypair.public_pem, signing_input, hashlib.sha256).digest())
        forged = (signing_input + b"." + sig).decode()
        with pytest.raises(AuthError) as exc:
            okta_validator(keypair).validate(forged)
        assert exc.value.status_code == 401

    def test_garbage_and_oversized_tokens_are_rejected(self, keypair):
        validator = okta_validator(keypair)
        for bad in ("not-a-jwt", "a.b.c", "x" * 9000):
            with pytest.raises(AuthError) as exc:
                validator.validate(bad)
            assert exc.value.status_code == 401

    def test_missing_token_is_401(self, keypair):
        with pytest.raises(AuthError) as exc:
            okta_validator(keypair).validate(None)
        assert (exc.value.status_code, exc.value.code) == (401, "missing_token")

    def test_jwks_outage_is_503_not_401(self, keypair):
        validator = okta_validator(keypair)
        validator._jwks = MagicMock()
        validator._jwks.get_signing_key_from_jwt.side_effect = jwt.PyJWKClientConnectionError("down")
        with pytest.raises(AuthError) as exc:
            validator.validate(rs256(keypair))
        assert (exc.value.status_code, exc.value.code) == (503, "jwks_unavailable")

    def test_unknown_signing_key_is_401(self, keypair):
        validator = okta_validator(keypair)
        validator._jwks = MagicMock()
        validator._jwks.get_signing_key_from_jwt.side_effect = jwt.PyJWKClientError("no kid")
        with pytest.raises(AuthError) as exc:
            validator.validate(rs256(keypair))
        assert exc.value.status_code == 401


class TestClaimMapping:
    @pytest.mark.parametrize("tenant", [None, "", "   ", ["a", "b"], 42])
    def test_missing_or_ambiguous_tenant_is_403(self, keypair, tenant):
        token = rs256(keypair, tenant_id=tenant) if tenant is not None else rs256(keypair, tenant_id=None)
        with pytest.raises(AuthError) as exc:
            okta_validator(keypair).validate(token)
        assert (exc.value.status_code, exc.value.code) == (403, "no_tenant")

    def test_scopes_accept_okta_list_or_space_string(self, keypair):
        as_list = okta_validator(keypair).validate(rs256(keypair, scp=["rag:query"]))
        as_string = okta_validator(keypair).validate(
            rs256(keypair, scp=None, scope="rag:query rag:web")
        )
        assert as_list.scopes == {"rag:query"}
        assert as_string.scopes == {"rag:query", "rag:web"}

    @pytest.mark.parametrize("raw,expected", [
        (9, 3), (-4, 0), ("2", 2), ("abc", 0), (True, 0), (None, 0), ([3], 0), (1.5, 0),
    ])
    def test_clearance_is_clamped_and_defaults_low(self, keypair, raw, expected):
        token = rs256(keypair, clearance=raw)
        assert okta_validator(keypair).validate(token).clearance == expected

    def test_groups_string_and_garbage(self, keypair):
        assert okta_validator(keypair).validate(rs256(keypair, groups="a b")).groups == {"a", "b"}
        assert okta_validator(keypair).validate(rs256(keypair, groups=[1, None, "ok"])).groups == {"ok"}


class TestDevModes:
    def test_dev_token_round_trip(self):
        validator = TokenValidator(make_settings(auth_mode="dev_token"))
        token = jwt.encode(claims(iss="dev-issuer"), DEV_SECRET, algorithm="HS256")
        assert validator.validate(token).tenant_id == "equities"

    def test_dev_token_wrong_issuer_rejected(self):
        validator = TokenValidator(make_settings(auth_mode="dev_token"))
        token = jwt.encode(claims(iss="somebody"), DEV_SECRET, algorithm="HS256")
        with pytest.raises(AuthError):
            validator.validate(token)

    def test_disabled_mode_returns_fixed_dev_principal(self):
        principal = TokenValidator(make_settings(auth_mode="disabled")).validate(None)
        assert principal.tenant_id == "dev" and principal.clearance == 3


class TestStartupValidation:
    @pytest.mark.parametrize("overrides", [
        dict(auth_mode="disabled", app_env="prod", audit_hmac_key="k"),
        dict(auth_mode="dev_token", app_env="staging", audit_hmac_key="k"),
        dict(auth_mode="bogus"),
        dict(auth_mode="okta", okta_issuer=""),
        dict(auth_mode="okta", okta_issuer="http://acme.okta.com/oauth2/default"),
        dict(auth_mode="okta", okta_jwks_url="http://acme.okta.com/keys"),
        dict(auth_mode="okta", okta_audience=""),
        dict(auth_mode="dev_token", dev_jwt_secret="short"),
        dict(auth_mode="okta", app_env="prod", audit_hmac_key=""),
    ])
    def test_unsafe_config_refuses_to_start(self, overrides):
        with pytest.raises(RuntimeError):
            validate_auth_config(make_settings(**overrides))

    def test_safe_configs_pass(self):
        validate_auth_config(make_settings(auth_mode="disabled"))
        validate_auth_config(make_settings(auth_mode="okta", app_env="prod", audit_hmac_key="secret"))


# ── Tenant policy ────────────────────────────────────────────────────────────

class TestTenantPolicy:
    def write(self, tmp_path, text):
        path = tmp_path / "policy.yaml"
        path.write_text(text)
        return path

    def test_unknown_tenant_gets_most_restrictive_policy(self):
        policy = TenantPolicyStore().for_tenant("nobody")
        assert policy == RESTRICTIVE
        assert not policy.allow_web_search and not policy.allow_url_ingest

    def test_missing_file_is_restrictive(self, tmp_path):
        assert TenantPolicyStore.load(tmp_path / "nope.yaml").for_tenant("x") == RESTRICTIVE

    def test_tenants_inherit_from_default_then_override(self, tmp_path):
        store = TenantPolicyStore.load(self.write(tmp_path, """
default: {max_classification: 2, ungrounded_answer_policy: warn}
tenants:
  eq: {allow_web_search: true, url_allowlist: [SEC.gov.]}
"""))
        eq = store.for_tenant("eq")
        assert eq.allow_web_search is True
        assert eq.max_classification == 2 and eq.ungrounded_answer_policy == "warn"
        assert eq.url_allowlist == ("sec.gov",)
        assert store.for_tenant("other").allow_web_search is False

    @pytest.mark.parametrize("body", [
        "tenants:\n  a: {allow_web_search: 'yes'}",
        "tenants:\n  a: {max_classification: 9}",
        "tenants:\n  a: {max_classification: true}",
        "tenants:\n  a: {ungrounded_answer_policy: ignore}",
        "tenants:\n  a: {typo_key: 1}",
        "tenants:\n  a: {url_allowlist: sec.gov}",
        "surprise: 1",
        "- just\n- a list",
        "tenants: [a, b]",
        "default: {",
    ])
    def test_malformed_policy_files_fail_loudly(self, tmp_path, body):
        with pytest.raises(PolicyError):
            TenantPolicyStore.load(self.write(tmp_path, body))


class TestWebSearchDecision:
    def principal(self, *scopes):
        return Principal(user_id="u", tenant_id="t", scopes=frozenset(scopes))

    def decide(self, *, tenant=True, scope=True, flag=None, globally=True):
        return web_search_decision(
            policy=TenantPolicy(allow_web_search=tenant),
            principal=self.principal(*(["rag:web"] if scope else [])),
            request_flag=flag,
            global_enabled=globally,
        )

    def test_all_layers_allow(self):
        assert self.decide() == (True, None)

    def test_tenant_ceiling_beats_user_scope(self):
        assert self.decide(tenant=False, scope=True) == (False, "tenant_policy")

    def test_user_without_scope_is_denied_even_if_tenant_allows(self):
        assert self.decide(tenant=True, scope=False) == (False, "missing_scope")

    def test_request_can_only_opt_out(self):
        assert self.decide(flag=False) == (False, "request_opt_out")
        assert self.decide(tenant=False, flag=True) == (False, "tenant_policy")
        assert self.decide(scope=False, flag=True) == (False, "missing_scope")
        assert self.decide(globally=False, flag=True) == (False, "global_disabled")


class TestUrlHostAllowed:
    policy = TenantPolicy(allow_url_ingest=True, url_allowlist=("sec.gov",))

    @pytest.mark.parametrize("url", [
        "https://sec.gov/a", "https://www.sec.gov/Archives/x", "https://SEC.GOV./x",
    ])
    def test_allowed(self, url):
        assert url_host_allowed(self.policy, url)

    @pytest.mark.parametrize("url", [
        "https://evilsec.gov/a",              # suffix without a dot boundary
        "https://sec.gov.evil.com/a",         # allowed name as a subdomain label
        "https://sec.gov@evil.com/a",         # userinfo trick: real host is evil.com
        "https://user:pw@sec.gov/a",          # credentials in URL
        "https://evil.com/?u=sec.gov",
        "not a url", "", "https://",
    ])
    def test_denied(self, url):
        assert not url_host_allowed(self.policy, url)

    def test_disabled_or_empty_allowlist_denies_everything(self):
        assert not url_host_allowed(TenantPolicy(allow_url_ingest=False, url_allowlist=("sec.gov",)), "https://sec.gov/")
        assert not url_host_allowed(TenantPolicy(allow_url_ingest=True), "https://sec.gov/")


# ── Endpoints ────────────────────────────────────────────────────────────────

@pytest.fixture
def api(audit_log):
    """TestClient with dev-token auth and a tenant policy store; no lifespan."""
    from online_pipeline.api import gateway
    from online_pipeline.auth import deps

    cfg = make_settings(auth_mode="dev_token")
    validator = TokenValidator(cfg)
    store = TenantPolicyStore(tenants={
        "open": TenantPolicy(allow_url_ingest=True, url_allowlist=("sec.gov",), allow_web_search=True),
        "closed": TenantPolicy(),
    })
    with patch.object(deps, "get_validator", return_value=validator), \
         patch.object(gateway, "_policies", store):
        yield SimpleNamespace(client=TestClient(gateway.app), gateway=gateway, audit=audit_log)


def bearer(tenant="open", scopes=("rag:query", "rag:ingest", "rag:web"), **extra) -> dict:
    token = jwt.encode(
        claims(iss="dev-issuer", tenant_id=tenant, scp=list(scopes), **extra),
        DEV_SECRET, algorithm="HS256",
    )
    return {"Authorization": f"Bearer {token}"}


def audit_events(audit) -> list[dict]:
    if not audit.path.exists():
        return []
    return [json.loads(line) for line in audit.path.read_text().splitlines()]


class TestEndpointAuthorization:
    def test_health_is_open(self, api):
        assert api.client.get("/health").status_code == 200

    @pytest.mark.parametrize("method,path,kwargs", [
        ("post", "/query", {"json": {"query": "q"}}),
        ("post", "/ingest/url", {"json": {"url": "https://sec.gov/x"}}),
        ("post", "/ingest/api", {"json": {"url": "https://sec.gov/x"}}),
        ("post", "/ingest", {"files": [("files", ("a.txt", b"hi", "text/plain"))]}),
    ])
    def test_every_route_requires_a_token(self, api, method, path, kwargs):
        response = getattr(api.client, method)(path, **kwargs)
        assert response.status_code == 401
        assert response.headers["www-authenticate"] == "Bearer"
        event = audit_events(api.audit)[-1]
        assert (event["action"], event["decision"], event["reason"]) == ("auth.fail", "deny", "missing_token")

    def test_bad_token_is_audited_without_the_token(self, api):
        response = api.client.post("/query", json={"query": "q"}, headers={"Authorization": "Bearer junk"})
        assert response.status_code == 401
        raw = api.audit.path.read_text()
        assert "junk" not in raw and "auth.fail" in raw

    def test_missing_scope_is_403_and_audited(self, api):
        response = api.client.post("/query", json={"query": "q"}, headers=bearer(scopes=("rag:ingest",)))
        assert response.status_code == 403
        event = audit_events(api.audit)[-1]
        assert (event["action"], event["reason"]) == ("authz.fail", "missing_scope")
        assert event["details"]["required_scope"] == "rag:query"

    def test_token_without_tenant_is_403(self, api):
        token = jwt.encode(claims(iss="dev-issuer", tenant_id=None), DEV_SECRET, algorithm="HS256")
        response = api.client.post("/query", json={"query": "q"}, headers={"Authorization": f"Bearer {token}"})
        assert response.status_code == 403


class TestUrlIngestPolicy:
    def test_tenant_without_permission_is_denied_before_any_fetch(self, api):
        with patch.object(api.gateway, "WebExtractor") as extractor:
            response = api.client.post(
                "/ingest/url", json={"url": "https://sec.gov/x"}, headers=bearer(tenant="closed")
            )
        assert response.status_code == 403
        extractor.return_value.extract.assert_not_called()
        event = audit_events(api.audit)[-1]
        assert (event["action"], event["reason"]) == ("policy.deny", "url_not_allowed")

    @pytest.mark.parametrize("url", [
        "https://sec.gov.evil.com/x", "https://evil.com/x", "https://sec.gov@evil.com/x",
        "http://169.254.169.254/latest/meta-data", "http://localhost:6333/",
    ])
    def test_hosts_outside_the_allowlist_are_denied(self, api, url):
        with patch.object(api.gateway, "WebExtractor") as extractor, \
             patch.object(api.gateway, "APIExtractor") as api_extractor:
            r1 = api.client.post("/ingest/url", json={"url": url}, headers=bearer())
            r2 = api.client.post("/ingest/api", json={"url": url}, headers=bearer())
        assert (r1.status_code, r2.status_code) == (403, 403)
        extractor.return_value.extract.assert_not_called()
        api_extractor.return_value.extract.assert_not_called()

    def test_allowed_host_is_ingested_and_audited_with_fingerprint_only(self, api):
        doc = {"text": "t", "metadata": {}}
        with patch.object(api.gateway, "WebExtractor") as extractor, \
             patch.object(api.gateway, "_index_document", return_value=api.gateway.IndexResult("doc1", 3)):
            extractor.return_value.extract.return_value = doc
            response = api.client.post(
                "/ingest/url", json={"url": "https://www.sec.gov/Archives/x"}, headers=bearer()
            )
        assert response.status_code == 200
        event = audit_events(api.audit)[-1]
        assert event["action"] == "ingest" and event["resource"] == {"doc_id": "doc1"}
        assert event["details"]["chunks"] == 3 and event["actor"] == "00u123"
        assert "sec.gov/Archives" not in api.audit.path.read_text()

    def test_file_ingest_audits_each_file_without_filename(self, api):
        with patch.object(api.gateway, "safe_extract_file") as extractor, \
             patch.object(api.gateway, "_index_document", return_value=api.gateway.IndexResult("d9", 2)):
            extractor.return_value = {"text": "t", "metadata": {}}
            response = api.client.post(
                "/ingest", headers=bearer(),
                files=[("files", ("secret-deal-memo.txt", b"hello", "text/plain")),
                       ("files", ("notes.exe", b"MZ", "application/octet-stream"))],
            )
        assert response.status_code == 200
        events = [e for e in audit_events(api.audit) if e["action"] == "ingest"]
        assert [e["details"]["status"] for e in events] == ["indexed", "error"]
        assert events[1]["reason"] == "unsupported_type"
        assert "secret-deal-memo" not in api.audit.path.read_text()


class TestQueryEndpoint:
    def run_query(self, api, tenant="open", body=None, **patches):
        g = api.gateway
        stubs = [
            patch.object(g, "_stage_rewrite", side_effect=lambda q: q),
            patch.object(g, "_stage_embed", return_value=[0.0]),
            patch.object(g, "_stage_retrieve", return_value=[
                {"text": "t", "metadata": {
                    "doc_id": "d1", "tenant_id": tenant, "owner_id": "00u123",
                    "acl_groups": [], "acl_public": False, "classification": 1}}
            ] * 4),
            patch.object(g, "_stage_rerank", side_effect=lambda q, c, k: c[:k]),
            patch.object(g, "_stage_build_context", return_value="ctx"),
            patch.object(g, "_stage_generate", return_value="the answer"),
            patch.object(g.settings, "answer_grading_enabled", False),
            patch.object(g, "_web_search", MagicMock(available=True)),
            patch.object(g, "_stage_web_search", return_value=[{"text": "w", "metadata": {}}]),
            patch.object(g, "_local_recall_is_weak", return_value=True),
        ]
        for stub in stubs:
            stub.start()
        try:
            return api.client.post(
                "/query", json=body or {"query": "what is the secret revenue figure?"},
                headers=bearer(tenant=tenant),
            )
        finally:
            for stub in reversed(stubs):
                stub.stop()

    def test_query_is_audited_without_query_text_or_answer(self, api):
        response = self.run_query(api)
        assert response.status_code == 200
        raw = api.audit.path.read_text()
        assert "secret revenue" not in raw and "the answer" not in raw
        event = audit_events(api.audit)[-1]
        assert event["action"] == "query" and event["tenant"] == "open"
        assert event["details"]["returned_doc_ids"] == ["d1"] * 4
        assert len(event["details"]["query_fp"]) == 64 and event["details"]["web_used"] is True

    def test_tenant_ceiling_blocks_web_search_and_records_policy_deny(self, api):
        response = self.run_query(api, tenant="closed")
        assert response.status_code == 200
        assert response.json()["web_search_used"] is False
        actions = [(e["action"], e["reason"]) for e in audit_events(api.audit)]
        assert ("policy.deny", "tenant_policy") in actions
        assert audit_events(api.audit)[-1]["details"]["web_denied"] == "tenant_policy"

    def test_request_opt_out_is_not_a_policy_denial(self, api):
        response = self.run_query(api, body={"query": "q", "web_search": False})
        assert response.json()["web_search_used"] is False
        assert all(e["action"] != "policy.deny" for e in audit_events(api.audit))

    def test_request_cannot_opt_in_past_the_tenant_ceiling(self, api):
        response = self.run_query(api, tenant="closed", body={"query": "q", "web_search": True})
        assert response.json()["web_search_used"] is False

    def test_failed_query_is_audited_as_error(self, api):
        with patch.object(api.gateway, "_answer_query", side_effect=RuntimeError("boom")):
            client = TestClient(api.gateway.app, raise_server_exceptions=False)
            response = client.post("/query", json={"query": "q"}, headers=bearer())
        assert response.status_code == 500
        event = audit_events(api.audit)[-1]
        assert (event["action"], event["decision"], event["reason"]) == ("query", "error", "RuntimeError")

    def test_audit_failure_fails_closed_with_503(self, api):
        from online_pipeline.audit.logger import AuditError, set_audit

        class Broken:
            def fingerprint(self, text):
                return "f" * 64

            def emit(self, *args, **kwargs):
                raise AuditError("disk full")

        set_audit(Broken())
        # The pipeline itself succeeds; only the audit write fails.
        ok = (api.gateway.QueryResponse(answer="a", sources=[]), {})
        client = TestClient(api.gateway.app, raise_server_exceptions=False)
        with patch.object(api.gateway, "_answer_query", return_value=ok):
            response = client.post("/query", json={"query": "q"}, headers=bearer())
        assert response.status_code == 503
        assert "audit" in response.json()["detail"].lower()
