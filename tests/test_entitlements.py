"""P3: entitlement-filtered retrieval against Qdrant's real filter engine.

The central property: the Qdrant pre-filter and the Python guard are two
implementations of one rule, so they must agree on every (document, principal)
pair. Everything else here exercises the paths around that rule.
"""

from __future__ import annotations

import json
import random
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import jwt
import pytest
from fastapi.testclient import TestClient

from config.settings import Settings
from online_pipeline.auth.acl import authorize, build_qdrant_filter, is_authorized
from online_pipeline.auth.policy import TenantPolicy, TenantPolicyStore
from online_pipeline.auth.principal import Principal
from online_pipeline.auth.tokens import TokenValidator
from online_pipeline.retrieval.hybrid_retriever import HybridRetriever
from vector_db.client import VectorDBClient

pytest.importorskip("rank_bm25")

TENANTS = ["eq", "ib", "legal"]
USERS = ["alice", "bob", "carol"]
GROUPS = ["fx", "rates", "credit"]
VECTOR = [1.0, 0.0, 0.0, 0.0]


def make_chunks(rng: random.Random, count: int = 240) -> list[dict]:
    chunks = []
    for i in range(count):
        chunks.append({
            "id": f"00000000-0000-0000-0000-{i:012d}",
            "text": f"shared common token alpha chunk number {i} body text",
            "payload": {
                "doc_id": f"doc{i}", "chunk_index": 0, "source": f"src{i % 7}",
                "tenant_id": rng.choice(TENANTS),
                "owner_id": rng.choice(USERS),
                "acl_groups": rng.sample(GROUPS, rng.randint(0, 2)),
                "acl_public": rng.random() < 0.2,
                "classification": rng.randint(0, 3),
                "quarantined": rng.random() < 0.12,
            },
        })
    return chunks


@pytest.fixture(scope="module")
def store():
    rng = random.Random(1234)
    client = VectorDBClient(url=":memory:", collection="t", dimension=4)
    client.ensure_collection()
    chunks = make_chunks(rng)
    client.upsert([
        {"id": c["id"], "vector": VECTOR, "payload": c["payload"], "text": c["text"]} for c in chunks
    ])
    return SimpleNamespace(client=client, chunks=chunks)


def random_principal(rng: random.Random) -> Principal:
    return Principal(
        user_id=rng.choice(USERS),
        tenant_id=rng.choice(TENANTS),
        groups=frozenset(rng.sample(GROUPS, rng.randint(0, 3))),
        clearance=rng.randint(0, 3),
    )


def expected_ids(chunks, principal) -> set[str]:
    return {c["id"] for c in chunks if is_authorized(c["payload"], principal)}


class TestFilterAndGuardAgree:
    def test_qdrant_filter_matches_the_python_rule_on_random_principals(self, store):
        rng = random.Random(99)
        checked_nonempty = 0
        for _ in range(300):
            principal = random_principal(rng)
            hits = store.client.search(VECTOR, top_k=1000, principal=principal)
            got = {h["id"] for h in hits}
            want = expected_ids(store.chunks, principal)
            assert got == want, f"{principal}: filter and guard disagree"
            checked_nonempty += bool(want)
        assert checked_nonempty > 150  # the matrix actually exercised visible data

    def test_quarantined_chunks_are_hidden_from_everyone_including_the_owner(self, store):
        quarantined = [c for c in store.chunks if c["payload"]["quarantined"]]
        assert len(quarantined) > 10
        for chunk in quarantined:
            owner = Principal(user_id=chunk["payload"]["owner_id"], tenant_id=chunk["payload"]["tenant_id"],
                              groups=frozenset(GROUPS), clearance=3)
            assert not is_authorized(chunk["payload"], owner)
            hits = {h["id"] for h in store.client.search(VECTOR, top_k=1000, principal=owner)}
            assert chunk["id"] not in hits

    def test_nothing_crosses_tenants(self, store):
        for tenant in TENANTS:
            principal = Principal(user_id="alice", tenant_id=tenant, groups=frozenset(GROUPS), clearance=3)
            for hit in store.client.search(VECTOR, top_k=1000, principal=principal):
                assert hit["metadata"]["tenant_id"] == tenant

    def test_clearance_is_a_hard_ceiling(self, store):
        principal = Principal(user_id="alice", tenant_id="eq", groups=frozenset(GROUPS), clearance=1)
        for hit in store.client.search(VECTOR, top_k=1000, principal=principal):
            assert hit["metadata"]["classification"] <= 1

    def test_principal_with_no_groups_still_works(self, store):
        principal = Principal(user_id="alice", tenant_id="eq", clearance=3)
        got = {h["id"] for h in store.client.search(VECTOR, top_k=1000, principal=principal)}
        assert got == expected_ids(store.chunks, principal)

    def test_points_without_acl_fields_are_invisible_to_everyone(self):
        client = VectorDBClient(url=":memory:", collection="legacy", dimension=4)
        client.ensure_collection()
        client.upsert([{"id": "00000000-0000-0000-0000-000000000001", "vector": VECTOR,
                        "payload": {"doc_id": "old"}, "text": "legacy chunk"}])
        principal = Principal(user_id="alice", tenant_id="eq", groups=frozenset(GROUPS), clearance=3)
        assert client.search(VECTOR, top_k=10, principal=principal) == []
        assert not is_authorized({"doc_id": "old"}, principal)

    def test_source_in_narrows_but_never_widens(self, store):
        principal = Principal(user_id="alice", tenant_id="eq", groups=frozenset({"fx"}), clearance=2)
        allowed = expected_ids(store.chunks, principal)
        hits = store.client.search(VECTOR, top_k=1000, principal=principal, source_in=["src1", "src2"])
        assert {h["id"] for h in hits} <= allowed
        assert all(h["metadata"]["source"] in {"src1", "src2"} for h in hits)
        # A source that only exists outside the principal's entitlement yields nothing.
        outside = next(c for c in store.chunks if c["id"] not in allowed)
        none = store.client.search(VECTOR, top_k=1000, principal=principal,
                                   source_in=[outside["payload"]["source"]])
        assert outside["id"] not in {h["id"] for h in none}
        assert {h["id"] for h in none} <= allowed

    def test_search_requires_a_principal(self, store):
        with pytest.raises(TypeError):
            store.client.search(VECTOR, top_k=5)  # type: ignore[call-arg]

    def test_guard_drops_exactly_what_the_rule_denies(self, store):
        rng = random.Random(7)
        docs = [{"text": c["text"], "metadata": c["payload"]} for c in store.chunks]
        for _ in range(50):
            principal = random_principal(rng)
            allowed, dropped = authorize(docs, principal)
            assert dropped == len(docs) - len(allowed)
            assert all(is_authorized(d["metadata"], principal) for d in allowed)


@pytest.fixture
def hybrid(store):
    retriever = HybridRetriever(vector_db=store.client)
    retriever.load_corpus(store.client.scroll_all())
    return retriever


class TestHybridRetrievalIsEntitled:
    def test_lexical_branch_is_live_and_per_tenant(self, hybrid):
        assert hybrid.lexical_enabled
        assert set(hybrid._bm25_by_tenant) == set(TENANTS)

    def test_hybrid_results_equal_the_entitled_set(self, store, hybrid):
        rng = random.Random(5)
        for _ in range(60):
            principal = random_principal(rng)
            results = hybrid.retrieve("alpha", VECTOR, principal, top_k=1000)
            got = {r["id"] for r in results}
            assert got == expected_ids(store.chunks, principal)

    def test_a_token_unique_to_another_tenant_finds_nothing(self, store):
        client = VectorDBClient(url=":memory:", collection="iso", dimension=4)
        client.ensure_collection()
        base = {"doc_id": "d", "chunk_index": 0, "owner_id": "x", "acl_groups": [], "acl_public": True, "classification": 0}
        points = [{"id": f"00000000-0000-0000-0000-{i:012d}", "vector": VECTOR,
                   "payload": {**base, "tenant_id": "ib"}, "text": f"filler {i} words here"} for i in range(30)]
        points.append({"id": "00000000-0000-0000-0000-0000000000ff", "vector": VECTOR,
                       "payload": {**base, "tenant_id": "ib"}, "text": "zzcodeword project falcon merger"})
        points += [{"id": f"00000000-0000-0000-0000-{100 + i:012d}", "vector": VECTOR,
                    "payload": {**base, "tenant_id": "eq"}, "text": f"unrelated eq text {i}"} for i in range(30)]
        client.upsert(points)
        retriever = HybridRetriever(vector_db=client)
        retriever.load_corpus(client.scroll_all())

        eq = Principal(user_id="u", tenant_id="eq", clearance=3)
        ib = Principal(user_id="u", tenant_id="ib", clearance=3)
        eq_results = retriever.retrieve("zzcodeword", VECTOR, eq, top_k=50)
        assert eq_results  # the vector branch still answers with eq's own documents
        assert all("zzcodeword" not in r["text"] for r in eq_results)
        assert all(r["metadata"]["tenant_id"] == "eq" for r in eq_results)
        hits = retriever.retrieve("zzcodeword", [], ib, top_k=5)
        assert hits and "zzcodeword" in hits[0]["text"]

    def test_other_tenants_documents_do_not_change_my_bm25_scores(self):
        def build(extra_tenant_docs: int):
            base = {"doc_id": "d", "chunk_index": 0, "owner_id": "x", "acl_groups": [], "acl_public": True, "classification": 0}
            corpus = [{"id": f"a{i}", "text": f"revenue guidance note {i}" if i else "rare keyword revenue",
                       "metadata": {**base, "tenant_id": "eq"}} for i in range(20)]
            corpus += [{"id": f"b{i}", "text": "revenue revenue revenue keyword", "metadata": {**base, "tenant_id": "ib"}}
                       for i in range(extra_tenant_docs)]
            retriever = HybridRetriever()
            retriever.load_corpus(corpus)
            eq = Principal(user_id="u", tenant_id="eq", clearance=3)
            return retriever._bm25_by_tenant["eq"].retrieve("keyword")[0]["bm25_score"], retriever, eq

        few, _, _ = build(0)
        many, retriever, eq = build(200)
        assert few == many  # IDF is computed over the tenant's own corpus only

    def test_revoking_access_takes_effect_before_the_index_is_rebuilt(self):
        client = VectorDBClient(url=":memory:", collection="rev", dimension=4)
        client.ensure_collection()
        pid = "00000000-0000-0000-0000-000000000001"
        acl = {"doc_id": "d", "chunk_index": 0, "tenant_id": "eq", "owner_id": "carol",
               "acl_groups": ["fx"], "acl_public": False, "classification": 1}
        client.upsert([{"id": pid, "vector": VECTOR, "payload": acl, "text": "unique lexicalmarker content"}]
                      + [{"id": f"00000000-0000-0000-0000-{i + 10:012d}", "vector": VECTOR,
                          "payload": {**acl, "doc_id": f"f{i}", "owner_id": "z", "acl_groups": ["other"]},
                          "text": f"filler words {i}"} for i in range(30)])
        retriever = HybridRetriever(vector_db=client)
        retriever.load_corpus(client.scroll_all())  # snapshot taken while access was allowed
        reader = Principal(user_id="bob", tenant_id="eq", groups=frozenset({"fx"}), clearance=2)
        assert retriever.retrieve("lexicalmarker", [], reader, top_k=5)

        client._client.set_payload("rev", payload={"acl_groups": ["legal-only"]}, points=[pid])
        assert retriever.retrieve("lexicalmarker", [], reader, top_k=5) == []

        client._client.set_payload("rev", payload={"acl_groups": ["fx"]}, points=[pid])
        assert retriever.retrieve("lexicalmarker", [], reader, top_k=5)  # restored
        client._client.delete("rev", points_selector=[pid])
        assert retriever.retrieve("lexicalmarker", [], reader, top_k=5) == []  # deleted

    def test_top_k_is_filled_with_permitted_documents(self):
        # 40 highly relevant chunks the caller may not read, 5 weaker ones they may.
        base = {"doc_id": "d", "chunk_index": 0, "tenant_id": "eq", "classification": 0}
        denied = {**base, "owner_id": "other", "acl_groups": ["secret"], "acl_public": False}
        allowed = {**base, "owner_id": "other", "acl_groups": [], "acl_public": True}
        def doc(prefix, i, text, acl):
            return {"id": f"{prefix}{i}", "text": text, "metadata": {**acl, "doc_id": f"{prefix}{i}"}}

        corpus = [doc("x", i, "ledger ledger ledger ledger", denied) for i in range(40)]
        corpus += [doc("y", i, f"ledger note {i} plus filler words", allowed) for i in range(5)]
        corpus += [doc("z", i, f"unrelated {i}", allowed) for i in range(20)]
        retriever = HybridRetriever()  # no store: isolates the pre-rank predicate
        retriever.load_corpus(corpus)

        principal = Principal(user_id="u", tenant_id="eq", clearance=3)
        results = retriever.retrieve("ledger", [], principal, top_k=3)
        assert len(results) == 3
        assert all(r["metadata"]["acl_public"] is True for r in results)

    def test_store_failure_during_revalidation_fails_closed(self, store):
        retriever = HybridRetriever(vector_db=store.client)
        retriever.load_corpus(store.client.scroll_all())
        principal = Principal(user_id="alice", tenant_id="eq", groups=frozenset(GROUPS), clearance=3)
        with patch.object(store.client, "fetch_payloads", side_effect=RuntimeError("down")):
            assert retriever.retrieve("alpha", [], principal, top_k=50) == []

    def test_corpus_chunks_without_tenant_are_never_indexed(self):
        retriever = HybridRetriever()
        retriever.load_corpus([{"id": "1", "text": "orphan chunk", "metadata": {"doc_id": "d"}}])
        assert not retriever.lexical_enabled


# ── Gateway: guard, web stamping, request model ──────────────────────────────

DEV_SECRET = "d" * 40


@pytest.fixture
def api(audit_log):
    from online_pipeline.api import gateway
    from online_pipeline.auth import deps

    cfg = Settings(_env_file=None, app_env="dev", auth_mode="dev_token",
                   dev_jwt_secret=DEV_SECRET, audit_hmac_key="k")
    policies = TenantPolicyStore(tenants={"eq": TenantPolicy(allow_web_search=True)})
    with patch.object(deps, "get_validator", return_value=TokenValidator(cfg)), \
         patch.object(gateway, "_policies", policies):
        yield SimpleNamespace(client=TestClient(gateway.app), gateway=gateway, audit=audit_log)


def bearer(tenant="eq", scopes=("rag:query", "rag:web"), groups=("fx",), clearance=2) -> dict:
    now = int(time.time())
    token = jwt.encode({
        "iss": "dev-issuer", "aud": "api://default", "sub": "alice", "iat": now, "exp": now + 600,
        "tenant_id": tenant, "groups": list(groups), "clearance": clearance, "scp": list(scopes),
    }, DEV_SECRET, algorithm="HS256")
    return {"Authorization": f"Bearer {token}"}


def chunk(doc_id, tenant="eq", **over):
    meta = {"doc_id": doc_id, "tenant_id": tenant, "owner_id": "alice", "acl_groups": [],
            "acl_public": False, "classification": 1, "source": "s"}
    meta.update(over)
    return {"text": f"text of {doc_id}", "metadata": meta}


class TestGatewayGuard:
    def run(self, api, retrieved, *, web=None, weak=False, body=None):
        g = api.gateway
        build_context = MagicMock(return_value="ctx")
        stubs = [
            patch.object(g, "_stage_rewrite", side_effect=lambda q: q),
            patch.object(g, "_stage_embed", return_value=[0.0]),
            patch.object(g, "_stage_retrieve", return_value=retrieved),
            patch.object(g, "_stage_rerank", side_effect=lambda q, c, k: c[:k]),
            patch.object(g, "_stage_build_context", build_context),
            patch.object(g, "_stage_generate", return_value="answer"),
            patch.object(g.settings, "answer_grading_enabled", False),
            patch.object(g, "_web_search", MagicMock(available=True)),
            patch.object(g, "_stage_web_search", return_value=web or []),
            patch.object(g, "_local_recall_is_weak", return_value=weak),
        ]
        [s.start() for s in stubs]
        try:
            response = api.client.post("/query", json=body or {"query": "q"}, headers=bearer())
        finally:
            [s.stop() for s in reversed(stubs)]
        return response, build_context

    def test_a_bypassed_filter_is_caught_audited_and_never_reaches_the_model(self, api):
        leaked = [chunk("ok"), chunk("other-tenant", tenant="ib"),
                  chunk("too-secret", classification=3), chunk("not-mine", owner_id="bob")]
        response, build_context = self.run(api, leaked)
        assert response.status_code == 200
        assert [s["doc_id"] for s in response.json()["sources"]] == ["ok"]
        assert [c["metadata"]["doc_id"] for c in build_context.call_args.args[0]] == ["ok"]
        events = [json.loads(l) for l in api.audit.path.read_text().splitlines()]
        guard = next(e for e in events if e["action"] == "acl.guard_dropped")
        assert guard["details"]["dropped"] >= 3 and guard["reason"] == "prefilter_bypass_suspected"
        query_event = next(e for e in events if e["action"] == "query")
        assert query_event["details"]["guard_dropped"] >= 3

    def test_the_final_guard_holds_even_if_a_later_stage_adds_a_chunk(self, api):
        g = api.gateway
        intruder = chunk("intruder", tenant="ib")
        build_context = MagicMock(return_value="ctx")
        stubs = [
            patch.object(g, "_stage_rewrite", side_effect=lambda q: q),
            patch.object(g, "_stage_embed", return_value=[0.0]),
            patch.object(g, "_stage_retrieve", return_value=[chunk("ok")]),
            # A misbehaving reranker (or future stage) injects a foreign chunk.
            patch.object(g, "_stage_rerank", side_effect=lambda q, c, k: c + [intruder]),
            patch.object(g, "_stage_build_context", build_context),
            patch.object(g, "_stage_generate", return_value="answer"),
            patch.object(g.settings, "answer_grading_enabled", False),
            patch.object(g, "_web_search", MagicMock(available=True)),
            patch.object(g, "_local_recall_is_weak", return_value=False),
        ]
        [s.start() for s in stubs]
        try:
            response = api.client.post("/query", json={"query": "q"}, headers=bearer())
        finally:
            [s.stop() for s in reversed(stubs)]
        assert [s["doc_id"] for s in response.json()["sources"]] == ["ok"]
        assert [c["metadata"]["doc_id"] for c in build_context.call_args.args[0]] == ["ok"]

    def test_clean_retrieval_raises_no_guard_event(self, api):
        self.run(api, [chunk("a"), chunk("b")])
        assert "acl.guard_dropped" not in api.audit.path.read_text()

    def test_web_results_get_an_explicit_acl_and_survive_the_guard(self, api):
        web = [{"text": "from the web", "metadata": {"source": "https://example.com", "origin": "tavily"}}]
        response, build_context = self.run(api, [chunk("a")], web=web, weak=True)
        assert response.json()["web_search_used"] is True
        web_chunk = next(c for c in build_context.call_args.args[0] if c["metadata"].get("origin") == "tavily")
        meta = web_chunk["metadata"]
        assert (meta["tenant_id"], meta["acl_public"], meta["classification"]) == ("eq", True, 0)

    def test_client_supplied_metadata_filter_is_rejected(self, api):
        response = api.client.post("/query", headers=bearer(),
                                   json={"query": "q", "metadata_filter": {"must": []}})
        assert response.status_code == 422

    def test_source_in_is_accepted_by_the_endpoint(self, api):
        response, _ = self.run(api, [chunk("a")], body={"query": "q", "source_in": ["s"]})
        assert response.status_code == 200

    def test_retrieve_stage_forwards_the_principal_and_source_filter(self, api):
        g = api.gateway
        principal = Principal(user_id="u", tenant_id="eq", clearance=1)
        retriever = MagicMock()
        retriever.retrieve.return_value = []
        with patch.object(g, "_retriever", retriever):
            g._stage_retrieve("text", [0.1], principal, ["s1"])
        kwargs = retriever.retrieve.call_args.kwargs
        assert kwargs["principal"] is principal and kwargs["source_in"] == ["s1"]
        assert "metadata_filter" not in kwargs


class TestFilterConstruction:
    def test_filter_has_tenant_clearance_and_access_branches(self):
        f = build_qdrant_filter(Principal(user_id="u", tenant_id="t", groups=frozenset({"g1"}), clearance=2))
        keys = [getattr(c, "key", None) for c in f.must]
        assert keys[:2] == ["tenant_id", "classification"]
        access = f.must[2]
        assert {c.key for c in access.should} == {"owner_id", "acl_public", "acl_groups"}

    def test_empty_group_list_is_omitted_not_sent_as_an_empty_match(self):
        f = build_qdrant_filter(Principal(user_id="u", tenant_id="t"))
        assert {c.key for c in f.must[2].should} == {"owner_id", "acl_public"}
