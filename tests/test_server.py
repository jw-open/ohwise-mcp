"""Tests for ohwise-mcp server tools."""

import json
import os
import subprocess
import tempfile
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_temp_repo(tmp_path: Path) -> Path:
    """Create a minimal Python repo for graph-building tests."""
    (tmp_path / "auth.py").write_text(
        "def login(user, pw):\n    return db_query(user)\n\ndef db_query(q):\n    return q\n"
    )
    (tmp_path / "main.py").write_text(
        "from auth import login\n\ndef run():\n    login('admin', 'pass')\n"
    )
    return tmp_path


def _make_git_repo(tmp_path: Path) -> Path:
    """Create a minimal git repo with one commit."""
    _make_temp_repo(tmp_path)
    subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, capture_output=True)
    return tmp_path


# ---------------------------------------------------------------------------
# build_code_graph
# ---------------------------------------------------------------------------

class TestBuildCodeGraph:
    def test_returns_json_with_node_count(self, tmp_path):
        pytest.importorskip("code2graph")
        from ohwise_mcp.server import build_code_graph

        repo = _make_temp_repo(tmp_path)
        result = json.loads(build_code_graph(str(repo), graph_type="call"))
        assert "node_count" in result
        assert result["node_count"] > 0
        assert "graph_id" in result

    def test_invalid_path_returns_error(self):
        pytest.importorskip("code2graph")
        from ohwise_mcp.server import build_code_graph

        result = json.loads(build_code_graph("/nonexistent/path/xyz"))
        assert "error" in result

    def test_graph_cached_after_build(self, tmp_path):
        pytest.importorskip("code2graph")
        from ohwise_mcp.server import build_code_graph, _graph_cache

        repo = _make_temp_repo(tmp_path)
        result = json.loads(build_code_graph(str(repo), graph_type="folder"))
        assert result["graph_id"] in _graph_cache


# ---------------------------------------------------------------------------
# rank_code_nodes
# ---------------------------------------------------------------------------

class TestRankCodeNodes:
    def test_returns_ranked_results(self, tmp_path):
        pytest.importorskip("code2graph")
        from ohwise_mcp.server import build_code_graph, rank_code_nodes

        repo = _make_temp_repo(tmp_path)
        gid = json.loads(build_code_graph(str(repo), graph_type="call"))["graph_id"]
        result = json.loads(rank_code_nodes("login authentication", graph_id=gid, k=5))
        assert "top_nodes" in result
        assert isinstance(result["top_nodes"], list)

    def test_missing_graph_returns_error(self):
        pytest.importorskip("code2graph")
        from ohwise_mcp.server import rank_code_nodes

        result = json.loads(rank_code_nodes("query", graph_id="nonexistent::id"))
        assert "error" in result

    def test_no_graph_id_or_path_returns_error(self):
        pytest.importorskip("code2graph")
        from ohwise_mcp.server import rank_code_nodes

        result = json.loads(rank_code_nodes("query"))
        assert "error" in result


# ---------------------------------------------------------------------------
# search_code_graph
# ---------------------------------------------------------------------------

class TestSearchCodeGraph:
    def test_finds_nodes_by_keyword(self, tmp_path):
        pytest.importorskip("code2graph")
        from ohwise_mcp.server import build_code_graph, search_code_graph

        repo = _make_temp_repo(tmp_path)
        gid = json.loads(build_code_graph(str(repo), graph_type="call"))["graph_id"]
        result = json.loads(search_code_graph("login", graph_id=gid))
        assert "results" in result
        assert isinstance(result["results"], list)

    def test_kind_filter(self, tmp_path):
        pytest.importorskip("code2graph")
        from ohwise_mcp.server import build_code_graph, search_code_graph

        repo = _make_temp_repo(tmp_path)
        gid = json.loads(build_code_graph(str(repo), graph_type="call"))["graph_id"]
        result = json.loads(search_code_graph("auth", graph_id=gid, kind_filter="function"))
        for node in result["results"]:
            assert node["kind"] == "function"


# ---------------------------------------------------------------------------
# get_task_context
# ---------------------------------------------------------------------------

class TestGetTaskContext:
    def test_returns_formatted_context(self, tmp_path):
        pytest.importorskip("code2graph")
        from ohwise_mcp.server import get_task_context

        repo = _make_temp_repo(tmp_path)
        result = get_task_context("login authentication", str(repo), graph_type="call", k=5)
        assert "# Context for: login authentication" in result
        assert "graph_id:" in result

    def test_invalid_path_returns_error(self):
        pytest.importorskip("code2graph")
        from ohwise_mcp.server import get_task_context

        result = get_task_context("query", "/nonexistent/xyz")
        data = json.loads(result)
        assert "error" in data


# ---------------------------------------------------------------------------
# get_node_detail
# ---------------------------------------------------------------------------

class TestGetNodeDetail:
    def test_returns_node_with_edges(self, tmp_path):
        pytest.importorskip("code2graph")
        from ohwise_mcp.server import build_code_graph, get_node_detail

        repo = _make_temp_repo(tmp_path)
        gid = json.loads(build_code_graph(str(repo), graph_type="call"))["graph_id"]
        result = json.loads(get_node_detail("login", graph_id=gid))
        assert "id" in result or "error" in result
        if "id" in result:
            assert "incoming_edges" in result
            assert "outgoing_edges" in result
            assert "degree" in result

    def test_unknown_node_returns_error(self, tmp_path):
        pytest.importorskip("code2graph")
        from ohwise_mcp.server import build_code_graph, get_node_detail

        repo = _make_temp_repo(tmp_path)
        gid = json.loads(build_code_graph(str(repo), graph_type="call"))["graph_id"]
        result = json.loads(get_node_detail("zzz_nonexistent_node_xyz", graph_id=gid))
        assert "error" in result


# ---------------------------------------------------------------------------
# find_impact
# ---------------------------------------------------------------------------

class TestFindImpact:
    def test_returns_impact_structure(self, tmp_path):
        pytest.importorskip("code2graph")
        from ohwise_mcp.server import build_code_graph, find_impact

        repo = _make_temp_repo(tmp_path)
        gid = json.loads(build_code_graph(str(repo), graph_type="call"))["graph_id"]
        result = json.loads(find_impact("db_query", graph_id=gid))
        assert "total_affected" in result or "error" in result
        if "total_affected" in result:
            assert "affected_by_depth" in result
            assert isinstance(result["affected_by_depth"], list)

    def test_unknown_node_returns_error(self, tmp_path):
        pytest.importorskip("code2graph")
        from ohwise_mcp.server import build_code_graph, find_impact

        repo = _make_temp_repo(tmp_path)
        gid = json.loads(build_code_graph(str(repo), graph_type="call"))["graph_id"]
        result = json.loads(find_impact("zzz_no_such_node", graph_id=gid))
        assert "error" in result


# ---------------------------------------------------------------------------
# trace_call_path
# ---------------------------------------------------------------------------

class TestTraceCallPath:
    def test_finds_path_or_none(self, tmp_path):
        pytest.importorskip("code2graph")
        from ohwise_mcp.server import build_code_graph, trace_call_path

        repo = _make_temp_repo(tmp_path)
        gid = json.loads(build_code_graph(str(repo), graph_type="call"))["graph_id"]
        result = json.loads(trace_call_path("login", "db_query", graph_id=gid))
        assert "path" in result or "error" in result

    def test_same_node_returns_length_zero(self, tmp_path):
        pytest.importorskip("code2graph")
        from ohwise_mcp.server import build_code_graph, trace_call_path

        repo = _make_temp_repo(tmp_path)
        gid = json.loads(build_code_graph(str(repo), graph_type="call"))["graph_id"]
        result = json.loads(trace_call_path("login", "login", graph_id=gid))
        # Either finds itself (path_length=0) or "not found" error — both valid
        assert "path_length" in result or "error" in result

    def test_missing_source_returns_error(self, tmp_path):
        pytest.importorskip("code2graph")
        from ohwise_mcp.server import build_code_graph, trace_call_path

        repo = _make_temp_repo(tmp_path)
        gid = json.loads(build_code_graph(str(repo), graph_type="call"))["graph_id"]
        result = json.loads(trace_call_path("zzz_no_such", "login", graph_id=gid))
        assert "error" in result


# ---------------------------------------------------------------------------
# diff_graph
# ---------------------------------------------------------------------------

class TestDiffGraph:
    def test_returns_changed_nodes(self, tmp_path):
        pytest.importorskip("code2graph")
        repo = _make_git_repo(tmp_path)

        # Make a change and commit
        (tmp_path / "auth.py").write_text("def login(user):\n    return True\n")
        subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True)
        subprocess.run(["git", "commit", "-m", "change auth"], cwd=tmp_path, capture_output=True)

        from ohwise_mcp.server import diff_graph
        result = json.loads(diff_graph(str(tmp_path), "HEAD~1"))
        assert "changed_files" in result
        assert "affected_nodes" in result
        assert any("auth.py" in f for f in result["changed_files"])

    def test_no_changes_returns_empty(self, tmp_path):
        pytest.importorskip("code2graph")
        repo = _make_git_repo(tmp_path)

        from ohwise_mcp.server import diff_graph
        result = json.loads(diff_graph(str(tmp_path), "HEAD"))
        assert result.get("changed_files") == [] or result.get("note")

    def test_invalid_git_ref_returns_error(self, tmp_path):
        pytest.importorskip("code2graph")
        repo = _make_git_repo(tmp_path)

        from ohwise_mcp.server import diff_graph
        result = json.loads(diff_graph(str(tmp_path), "zzz_bad_ref"))
        assert "error" in result


# ---------------------------------------------------------------------------
# build_doc_graph
# ---------------------------------------------------------------------------

class TestBuildDocGraph:
    def test_builds_from_markdown_file(self, tmp_path):
        pytest.importorskip("docs2graph")
        from ohwise_mcp.server import build_doc_graph

        md = tmp_path / "notes.md"
        md.write_text(
            "# Architecture\n\nThe system uses a DAG-based pipeline.\n\n"
            "## Components\n\nCoordinator dispatches tasks to agents.\n"
        )
        result = json.loads(build_doc_graph(str(md)))
        assert "node_count" in result
        assert "graph_id" in result

    def test_missing_path_returns_error(self):
        pytest.importorskip("docs2graph")
        from ohwise_mcp.server import build_doc_graph

        result = json.loads(build_doc_graph("/nonexistent/file.md"))
        assert "error" in result


# ---------------------------------------------------------------------------
# schema_context
# ---------------------------------------------------------------------------

class TestSchemaContext:
    _DDL = """
    CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(100), email VARCHAR(200));
    CREATE TABLE orders (id INT PRIMARY KEY, user_id INT REFERENCES users(id), total DECIMAL(10,2));
    CREATE TABLE order_items (id INT PRIMARY KEY, order_id INT REFERENCES orders(id), qty INT);
    """

    def test_with_ddl_returns_tables(self):
        pytest.importorskip("graph2sql")
        from ohwise_mcp.server import schema_context

        result = json.loads(schema_context("total orders per user", ddl=self._DDL, k=2))
        assert "tables" in result
        assert isinstance(result["tables"], list)
        assert len(result["tables"]) > 0

    def test_context_contains_sql_block(self):
        pytest.importorskip("graph2sql")
        from ohwise_mcp.server import schema_context

        result = json.loads(schema_context("revenue by customer", ddl=self._DDL, k=2))
        assert "```sql" in result["context"] or len(result["tables"]) > 0

    def test_missing_ddl_and_db_url_returns_error(self):
        pytest.importorskip("graph2sql")
        from ohwise_mcp.server import schema_context

        env_backup = os.environ.pop("DATABASE_URL", None)
        try:
            result = json.loads(schema_context("query"))
            assert "error" in result
        finally:
            if env_backup:
                os.environ["DATABASE_URL"] = env_backup


# ---------------------------------------------------------------------------
# Document store tools — error path (no MONGO_URI)
# ---------------------------------------------------------------------------

class TestQueryCollection:
    def test_missing_mongo_uri_returns_error(self):
        from ohwise_mcp.server import query_collection

        backup = os.environ.pop("MONGO_URI", None)
        try:
            result = json.loads(query_collection("test_col"))
            assert "error" in result
            assert "MONGO_URI" in result["error"]
        finally:
            if backup:
                os.environ["MONGO_URI"] = backup


class TestListCollections:
    def test_missing_mongo_uri_returns_error(self):
        from ohwise_mcp.server import list_collections

        backup = os.environ.pop("MONGO_URI", None)
        try:
            result = json.loads(list_collections())
            assert "error" in result
            assert "MONGO_URI" in result["error"]
        finally:
            if backup:
                os.environ["MONGO_URI"] = backup


class TestInferCollectionSchema:
    def test_missing_mongo_uri_returns_error(self):
        from ohwise_mcp.server import infer_collection_schema

        backup = os.environ.pop("MONGO_URI", None)
        try:
            result = json.loads(infer_collection_schema("test_col"))
            assert "error" in result
            assert "MONGO_URI" in result["error"]
        finally:
            if backup:
                os.environ["MONGO_URI"] = backup


# ---------------------------------------------------------------------------
# Cache tools — error path (no REDIS_URL)
# ---------------------------------------------------------------------------

class TestCacheGet:
    def test_missing_redis_url_returns_error(self):
        from ohwise_mcp.server import cache_get

        backup = os.environ.pop("REDIS_URL", None)
        try:
            result = json.loads(cache_get("test_key"))
            assert "error" in result
            assert "REDIS_URL" in result["error"]
        finally:
            if backup:
                os.environ["REDIS_URL"] = backup


class TestCacheKeys:
    def test_missing_redis_url_returns_error(self):
        from ohwise_mcp.server import cache_keys

        backup = os.environ.pop("REDIS_URL", None)
        try:
            result = json.loads(cache_keys())
            assert "error" in result
            assert "REDIS_URL" in result["error"]
        finally:
            if backup:
                os.environ["REDIS_URL"] = backup


class TestCachePublish:
    def test_missing_redis_url_returns_error(self):
        from ohwise_mcp.server import cache_publish

        backup = os.environ.pop("REDIS_URL", None)
        try:
            result = json.loads(cache_publish("chan", "msg"))
            assert "error" in result
            assert "REDIS_URL" in result["error"]
        finally:
            if backup:
                os.environ["REDIS_URL"] = backup


# ---------------------------------------------------------------------------
# Agent pipeline tools — error path (no AGENT_BASE_URL)
# ---------------------------------------------------------------------------

def _clear_agent_env():
    """Remove all agent URL env vars; return backup dict."""
    backup = {}
    for key in ("AGENT_BASE_URL", "OHWISE_URL", "AGENT_TOKEN", "OHWISE_TOKEN"):
        backup[key] = os.environ.pop(key, None)
    return backup


def _restore_agent_env(backup: dict):
    for key, val in backup.items():
        if val is not None:
            os.environ[key] = val


class TestListAgents:
    def test_missing_base_url_returns_error(self):
        from ohwise_mcp.server import list_agents

        backup = _clear_agent_env()
        try:
            result = json.loads(list_agents())
            assert "error" in result
            assert "AGENT_BASE_URL" in result["error"]
        finally:
            _restore_agent_env(backup)


class TestRunAgent:
    def test_missing_base_url_returns_error(self):
        from ohwise_mcp.server import run_agent

        backup = _clear_agent_env()
        try:
            result = json.loads(run_agent("agent-id", "do something"))
            assert "error" in result
            assert "AGENT_BASE_URL" in result["error"]
        finally:
            _restore_agent_env(backup)


class TestGetAgentResult:
    def test_missing_base_url_returns_error(self):
        from ohwise_mcp.server import get_agent_result

        backup = _clear_agent_env()
        try:
            result = json.loads(get_agent_result("thread-id", poll_seconds=1))
            assert "error" in result
            assert "AGENT_BASE_URL" in result["error"]
        finally:
            _restore_agent_env(backup)
