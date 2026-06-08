"""Integration tests for ohwise-mcp OhWise API tools.

These tests hit the real OhWise backend and require a running backend plus
a PERSONAL_LAB_SECRET env var (or the backend's personal-token endpoint).

Run with:
    PERSONAL_LAB_SECRET=<secret> pytest tests/test_ohwise_integration.py -v

Skipped automatically when the backend is unreachable or secret is absent.
"""

import json
import os
import urllib.request
import uuid

import pytest

_OHWISE_URL = os.environ.get("OHWISE_URL", "http://localhost:8000")


# ---------------------------------------------------------------------------
# Session-scoped auth — gets a real JWT once for all integration tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def ohwise_token():
    secret = os.environ.get("PERSONAL_LAB_SECRET", "")
    if not secret:
        pytest.skip("PERSONAL_LAB_SECRET not set — skipping integration tests")
    try:
        data = json.dumps({"secret": secret}).encode()
        req = urllib.request.Request(
            f"{_OHWISE_URL}/api/personal-token",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            body = json.loads(resp.read())
            token = body.get("access_token", "")
            if not token:
                pytest.skip(f"personal-token returned no token: {body}")
            return token
    except Exception as exc:
        pytest.skip(f"OhWise backend not reachable at {_OHWISE_URL}: {exc}")


@pytest.fixture(autouse=True)
def inject_env(ohwise_token, monkeypatch):
    """Inject OHWISE_URL and OHWISE_TOKEN into every test's environment."""
    monkeypatch.setenv("OHWISE_URL", _OHWISE_URL)
    monkeypatch.setenv("OHWISE_TOKEN", ohwise_token)


def _api_get(token: str, path: str) -> dict:
    """Direct HTTP GET to the OhWise backend (bypasses MCP layer)."""
    req = urllib.request.Request(
        f"{_OHWISE_URL}{path}",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())


# ---------------------------------------------------------------------------
# list_agents
# ---------------------------------------------------------------------------

class TestListAgentsIntegration:
    def test_returns_no_error(self):
        from ohwise_mcp.server import list_agents

        result = json.loads(list_agents())
        assert "error" not in result, f"list_agents returned error: {result['error']}"

    def test_agents_have_agent_id(self):
        from ohwise_mcp.server import list_agents

        result = json.loads(list_agents())
        data = result.get("data", result)
        agents = data if isinstance(data, list) else []
        assert len(agents) > 0, "Expected at least one agent"
        for agent in agents:
            assert "agent_id" in agent, f"Missing agent_id: {agent}"


# ---------------------------------------------------------------------------
# knowledge_list
# ---------------------------------------------------------------------------

class TestKnowledgeListIntegration:
    def test_returns_items_list(self):
        from ohwise_mcp.server import knowledge_list

        result = json.loads(knowledge_list())
        assert "error" not in result, f"knowledge_list error: {result['error']}"
        assert "items" in result
        assert isinstance(result["items"], list)

    def test_response_has_pagination_fields(self):
        from ohwise_mcp.server import knowledge_list

        result = json.loads(knowledge_list(page=1, page_size=5))
        assert "total" in result
        assert result["page"] == 1

    def test_items_have_uuid_knowledge_id(self):
        from ohwise_mcp.server import knowledge_list

        result = json.loads(knowledge_list())
        for item in result["items"]:
            kid = item.get("knowledge_id", "")
            assert len(kid) == 36, f"knowledge_id is not a UUID: {kid!r}"


# ---------------------------------------------------------------------------
# knowledge_get
# ---------------------------------------------------------------------------

class TestKnowledgeGetIntegration:
    @pytest.fixture(scope="class")
    def first_knowledge_id(self, ohwise_token):
        body = _api_get(ohwise_token, "/api/knowledge?page=1&pageSize=1")
        items = body.get("data", [])
        if not items:
            pytest.skip("No knowledge bases found — create one first")
        return items[0]["knowledge_id"]

    def test_get_returns_correct_id(self, first_knowledge_id):
        from ohwise_mcp.server import knowledge_get

        result = json.loads(knowledge_get(first_knowledge_id))
        assert "error" not in result, f"knowledge_get error: {result['error']}"
        assert result["knowledge_id"] == first_knowledge_id

    def test_get_returns_graph_structure(self, first_knowledge_id):
        from ohwise_mcp.server import knowledge_get

        result = json.loads(knowledge_get(first_knowledge_id))
        assert "node_count" in result
        assert "edge_count" in result
        assert "graph" in result
        assert isinstance(result["node_count"], int)
        assert isinstance(result["edge_count"], int)

    def test_unknown_id_returns_error(self):
        from ohwise_mcp.server import knowledge_get

        result = json.loads(knowledge_get(str(uuid.uuid4())))
        assert "error" in result


# ---------------------------------------------------------------------------
# knowledge_query
# ---------------------------------------------------------------------------

class TestKnowledgeQueryIntegration:
    @pytest.fixture(scope="class")
    def knowledge_with_nodes(self, ohwise_token):
        body = _api_get(ohwise_token, "/api/knowledge?page=1&pageSize=20")
        for item in body.get("data", []):
            kid = item["knowledge_id"]
            kg_body = _api_get(ohwise_token, f"/api/knowledge/{kid}/graph")
            graph = kg_body.get("data") or {}
            if len(graph.get("nodes", [])) > 0:
                return kid
        pytest.skip("No knowledge base with nodes found")

    def test_query_returns_result_structure(self, knowledge_with_nodes):
        from ohwise_mcp.server import knowledge_query

        result = json.loads(knowledge_query(knowledge_with_nodes, "data", k=5))
        assert "error" not in result, f"knowledge_query error: {result['error']}"
        assert "top_nodes" in result
        assert "total_nodes" in result
        assert isinstance(result["top_nodes"], list)

    def test_query_respects_k_limit(self, knowledge_with_nodes):
        from ohwise_mcp.server import knowledge_query

        result = json.loads(knowledge_query(knowledge_with_nodes, "the", k=2))
        assert len(result["top_nodes"]) <= 2


# ---------------------------------------------------------------------------
# knowledge node CRUD roundtrip
# ---------------------------------------------------------------------------

class TestKnowledgeNodeRoundtrip:
    @pytest.fixture(scope="class")
    def target_knowledge_id(self, ohwise_token):
        body = _api_get(ohwise_token, "/api/knowledge?page=1&pageSize=1")
        items = body.get("data", [])
        if not items:
            pytest.skip("No knowledge bases found")
        return items[0]["knowledge_id"]

    def test_add_then_delete_node(self, target_knowledge_id):
        from ohwise_mcp.server import knowledge_add_node, knowledge_delete_node, knowledge_get

        label = f"integration-test-{uuid.uuid4().hex[:8]}"

        # Add node
        add_result = json.loads(
            knowledge_add_node(
                target_knowledge_id,
                label=label,
                attributes='{"test": true}',
                content="Temporary node — integration test",
            )
        )
        assert "error" not in add_result, f"add_node failed: {add_result}"
        node_id = add_result["node_id"]
        assert len(node_id) == 36, f"Expected UUID node_id, got: {node_id!r}"

        # Verify node appears in graph
        kg = json.loads(knowledge_get(target_knowledge_id))
        labels_in_graph = [n.get("label") for n in kg["graph"].get("nodes", [])]
        assert label in labels_in_graph, f"Node not found after add. Labels: {labels_in_graph}"

        # Delete node
        del_result = json.loads(knowledge_delete_node(target_knowledge_id, node_id))
        assert "error" not in del_result, f"delete_node failed: {del_result}"
        assert del_result["removed_nodes"] == 1

        # Verify removed
        kg_after = json.loads(knowledge_get(target_knowledge_id))
        labels_after = [n.get("label") for n in kg_after["graph"].get("nodes", [])]
        assert label not in labels_after, f"Node still present after delete: {labels_after}"
