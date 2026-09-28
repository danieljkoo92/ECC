"""MCP server and Claude Desktop installer tests."""

import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pytest

from quantlab import claude_desktop as cd
from quantlab import mcp_server as m


def run(coro):
    return asyncio.run(coro)


# --- tools --------------------------------------------------------------------------------------


def test_server_exposes_exactly_the_three_tools():
    names = sorted(t.name for t in run(m.server.list_tools()))
    assert names == ["get_report", "test_nba_betting_systems", "test_trading_strategies"]


def test_market_tool_returns_a_full_report_on_demo_data():
    md = run(m.test_trading_strategies(symbol="SIM", source="synthetic", include_control=False))
    assert "# Strategy search report - SIM" in md
    assert "Bottom line" in md and "The funnel" in md


def test_nba_tool_returns_a_full_report_on_the_demo_league():
    md = run(m.test_nba_betting_systems(demo=True, include_control=False))
    assert "# Betting system search - NBA (demo league)" in md
    assert "Bottom line" in md


def test_bad_inputs_get_plain_answers_not_crashes():
    assert "ticker" in run(m.test_trading_strategies(symbol="rm -rf /"))
    assert "YYYY-MM-DD" in run(m.test_trading_strategies(symbol="QQQ", start="last year"))
    assert "No run with job id" in run(m.get_report("nope"))
    msg = run(m.test_nba_betting_systems(first_season="1990-91"))
    assert msg.startswith("The run failed") and "2007-08" in msg


def test_season_range_needs_a_valid_span_of_four_or_more():
    assert m.season_range("2015-16", "2018-19") == ["2015-16", "2016-17", "2017-18", "2018-19"]
    with pytest.raises(ValueError):
        m.season_range("2015-16", "2016-17")


def test_slow_runs_hand_back_a_job_id_that_get_report_collects(monkeypatch):
    def slow():
        time.sleep(0.5)
        return "the report"

    monkeypatch.setattr(m, "WAIT_SECONDS", 0.01)
    job_id = m._start("slow test job", slow)
    first = run(m._wait(job_id))
    assert "Still running" in first and job_id in first
    monkeypatch.setattr(m, "WAIT_SECONDS", 10.0)
    assert run(m.get_report(job_id)) == "the report"


def test_a_failing_run_reports_its_error():
    def broken():
        raise ValueError("no data for that symbol")

    job_id = m._start("broken job", broken)
    assert run(m._wait(job_id)) == "The run failed: no data for that symbol"


# --- the wire -----------------------------------------------------------------------------------


def _rpc(proc, message):
    proc.stdin.write(json.dumps(message) + "\n")
    proc.stdin.flush()


def _read(proc, want_id, timeout=60):
    deadline = time.time() + timeout
    while time.time() < deadline:
        line = proc.stdout.readline()
        if not line:
            break
        msg = json.loads(line)
        if msg.get("id") == want_id:
            return msg
    raise AssertionError(f"no response with id {want_id}")


def test_stdio_session_lists_tools_and_answers_a_call(tmp_path):
    env = dict(os.environ, QUANTLAB_CACHE=str(tmp_path), PYTHONPATH=ROOT)
    proc = subprocess.Popen([sys.executable, "-m", "quantlab.mcp_server"], cwd=ROOT, env=env, text=True,
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    try:
        _rpc(proc, {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
            "protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "pytest", "version": "0"}}})
        init = _read(proc, 1)
        assert init["result"]["serverInfo"]["name"] == "ai-quant-lab"
        _rpc(proc, {"jsonrpc": "2.0", "method": "notifications/initialized"})
        _rpc(proc, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        names = sorted(t["name"] for t in _read(proc, 2)["result"]["tools"])
        assert names == ["get_report", "test_nba_betting_systems", "test_trading_strategies"]
        _rpc(proc, {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                    "params": {"name": "get_report", "arguments": {"job_id": "abc"}}})
        text = _read(proc, 3)["result"]["content"][0]["text"]
        assert "No run with job id" in text
    finally:
        proc.stdin.close()
        proc.terminate()
        proc.wait(timeout=10)


def test_selftest_command_passes():
    out = subprocess.run([sys.executable, "-m", "quantlab.mcp_server", "--selftest"], cwd=ROOT, text=True,
                         capture_output=True, timeout=180, env=dict(os.environ, PYTHONPATH=ROOT))
    assert out.returncode == 0 and "selftest OK" in out.stdout


# --- Claude Desktop config ----------------------------------------------------------------------

EXISTING = {
    "mcpServers": {"smart-model-router": {"command": "node", "args": ["C:\\router\\index.js"]}},
    "preferences": {"theme": "dark"},
}


def _win_env(tmp_path):
    return {"APPDATA": str(tmp_path / "Roaming"), "LOCALAPPDATA": str(tmp_path / "Local")}


def _write(path: Path, text: str, bom=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((b"\xef\xbb\xbf" if bom else b"") + text.encode("utf-8"))


def test_install_keeps_existing_servers_and_settings_and_backs_up(tmp_path):
    env = _win_env(tmp_path)
    cfg = tmp_path / "Roaming" / "Claude" / cd.CONFIG_FILE
    _write(cfg, json.dumps(EXISTING))
    [result] = cd.install("C:\\Users\\d\\.local\\bin\\uvx.exe", env=env, platform="win32", home=tmp_path)
    data = json.loads(cfg.read_text(encoding="utf-8"))
    assert data["mcpServers"]["smart-model-router"] == EXISTING["mcpServers"]["smart-model-router"]
    assert data["preferences"] == {"theme": "dark"}
    assert data["mcpServers"]["ai-quant-lab"] == {
        "command": "C:\\Users\\d\\.local\\bin\\uvx.exe", "args": ["--from", cd.DEFAULT_SOURCE, "quantlab-mcp"]}
    assert result.other_servers == ["smart-model-router"] and not result.created
    assert json.loads(result.backup.read_text(encoding="utf-8")) == EXISTING


def test_install_is_idempotent(tmp_path):
    env = _win_env(tmp_path)
    cfg = tmp_path / "Roaming" / "Claude" / cd.CONFIG_FILE
    _write(cfg, json.dumps(EXISTING))
    cd.install("uvx.exe", env=env, platform="win32", home=tmp_path)
    [second] = cd.install("uvx.exe", env=env, platform="win32", home=tmp_path)
    data = json.loads(cfg.read_text(encoding="utf-8"))
    assert list(data["mcpServers"]) == ["smart-model-router", "ai-quant-lab"]
    assert second.replaced_existing_entry


def test_install_reads_a_notepad_file_with_a_byte_order_mark(tmp_path):
    env = _win_env(tmp_path)
    cfg = tmp_path / "Roaming" / "Claude" / cd.CONFIG_FILE
    _write(cfg, json.dumps(EXISTING), bom=True)
    cd.install("uvx.exe", env=env, platform="win32", home=tmp_path)
    assert "smart-model-router" in json.loads(cfg.read_text(encoding="utf-8"))["mcpServers"]


def test_invalid_json_is_left_untouched_with_no_backup(tmp_path):
    env = _win_env(tmp_path)
    cfg = tmp_path / "Roaming" / "Claude" / cd.CONFIG_FILE
    _write(cfg, '{"mcpServers": {"smart-model-router": {,}}')
    with pytest.raises(cd.ConfigError):
        cd.install("uvx.exe", env=env, platform="win32", home=tmp_path)
    assert cfg.read_text(encoding="utf-8") == '{"mcpServers": {"smart-model-router": {,}}'
    assert not list(cfg.parent.glob("*.bak-*"))


def test_non_object_mcp_servers_is_refused(tmp_path):
    env = _win_env(tmp_path)
    _write(tmp_path / "Roaming" / "Claude" / cd.CONFIG_FILE, '{"mcpServers": []}')
    with pytest.raises(cd.ConfigError):
        cd.install("uvx.exe", env=env, platform="win32", home=tmp_path)


def test_creates_the_config_when_claude_has_none_yet(tmp_path):
    env = _win_env(tmp_path)
    [result] = cd.install("uvx.exe", env=env, platform="win32", home=tmp_path)
    assert result.created and result.backup is None
    assert json.loads(result.path.read_text(encoding="utf-8"))["mcpServers"]["ai-quant-lab"]["command"] == "uvx.exe"


def test_microsoft_store_install_is_found_and_updated_too(tmp_path):
    env = _win_env(tmp_path)
    standard = tmp_path / "Roaming" / "Claude" / cd.CONFIG_FILE
    store = tmp_path / "Local" / "Packages" / "Claude_pzs8sxrjxfjjc" / "LocalCache" / "Roaming" / "Claude" / cd.CONFIG_FILE
    _write(standard, json.dumps(EXISTING))
    _write(store, json.dumps(EXISTING))
    results = cd.install("uvx.exe", env=env, platform="win32", home=tmp_path)
    assert {r.path for r in results} == {standard, store}
    for p in (standard, store):
        assert set(json.loads(p.read_text(encoding="utf-8"))["mcpServers"]) == {"smart-model-router", "ai-quant-lab"}


def test_one_unreadable_file_stops_all_writes(tmp_path):
    env = _win_env(tmp_path)
    standard = tmp_path / "Roaming" / "Claude" / cd.CONFIG_FILE
    store = tmp_path / "Local" / "Packages" / "Claude_x" / "LocalCache" / "Roaming" / "Claude" / cd.CONFIG_FILE
    _write(standard, json.dumps(EXISTING))
    _write(store, "not json")
    with pytest.raises(cd.ConfigError):
        cd.install("uvx.exe", env=env, platform="win32", home=tmp_path)
    assert json.loads(standard.read_text(encoding="utf-8")) == EXISTING


def test_mac_and_linux_paths():
    home = Path("/home/x")
    assert cd.config_paths(env={}, platform="darwin", home=home)[0] == \
        home / "Library" / "Application Support" / "Claude" / cd.CONFIG_FILE
    assert cd.config_paths(env={}, platform="linux", home=home)[0] == home / ".config" / "Claude" / cd.CONFIG_FILE


def test_install_command_line_reports_what_it_kept(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("APPDATA", str(tmp_path / "Roaming"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "Local"))
    _write(tmp_path / "Roaming" / "Claude" / cd.CONFIG_FILE, json.dumps(EXISTING))
    assert m.main(["--install-claude-desktop", "--uvx-path", "C:\\uv\\uvx.exe"]) == 0
    out = capsys.readouterr().out
    assert "Added ai-quant-lab" in out and "smart-model-router" in out and "backup" in out
