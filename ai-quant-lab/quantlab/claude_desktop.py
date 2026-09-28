"""Register the ai-quant-lab MCP server with Claude Desktop.

Claude Desktop reads its local MCP servers from `claude_desktop_config.json`.
This module adds one entry to that file without touching anything else in it:
every other server stays, every other setting stays, a timestamped backup is
written first, and a file that is not valid JSON is left alone rather than
overwritten.

On Windows the file normally lives in %APPDATA%\\Claude. The Microsoft Store
build of Claude keeps its own copy under %LOCALAPPDATA%\\Packages\\Claude_*,
so both locations are checked and every one that exists is updated.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional

SERVER_NAME = "ai-quant-lab"
DEFAULT_SOURCE = "https://github.com/danieljkoo92/ECC/archive/refs/heads/main.zip#subdirectory=ai-quant-lab"
CONFIG_FILE = "claude_desktop_config.json"


class ConfigError(RuntimeError):
    """The existing config could not be read safely, so nothing was written."""


@dataclass
class InstallResult:
    path: Path
    created: bool
    replaced_existing_entry: bool
    backup: Optional[Path]
    other_servers: List[str]


def config_paths(env: Optional[Mapping[str, str]] = None, platform: Optional[str] = None,
                 home: Optional[Path] = None) -> List[Path]:
    """Every Claude Desktop config file to update, most standard first.

    Returns the files that already exist. If none exists yet, returns the one
    location a fresh Claude install would use, so it can be created.
    """
    env = dict(os.environ if env is None else env)
    platform = platform or sys.platform
    home = home or Path.home()
    candidates: List[Path] = []
    if platform.startswith("win"):
        appdata = env.get("APPDATA") or str(home / "AppData" / "Roaming")
        candidates.append(Path(appdata) / "Claude" / CONFIG_FILE)
        local = env.get("LOCALAPPDATA") or str(home / "AppData" / "Local")
        packages = Path(local) / "Packages"
        if packages.is_dir():
            for pkg in sorted(packages.glob("Claude_*")):
                candidates.append(pkg / "LocalCache" / "Roaming" / "Claude" / CONFIG_FILE)
    elif platform == "darwin":
        candidates.append(home / "Library" / "Application Support" / "Claude" / CONFIG_FILE)
    else:
        config_home = env.get("XDG_CONFIG_HOME") or str(home / ".config")
        candidates.append(Path(config_home) / "Claude" / CONFIG_FILE)
    existing = [p for p in candidates if p.exists()]
    return existing or candidates[:1]


def server_entry(uvx_path: str, source: str = DEFAULT_SOURCE) -> Dict[str, object]:
    """The config entry that makes Claude launch the server through uvx."""
    return {"command": str(uvx_path), "args": ["--from", source, "quantlab-mcp"]}


def _read_config(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    # utf-8-sig: Notepad on Windows saves JSON with a byte-order mark.
    text = path.read_text(encoding="utf-8-sig")
    if not text.strip():
        return {}
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{path} is not valid JSON ({exc}); left it unchanged") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path} does not contain a JSON object; left it unchanged")
    servers = data.get("mcpServers", {})
    if not isinstance(servers, dict):
        raise ConfigError(f"'mcpServers' in {path} is not an object; left it unchanged")
    return data


def install_into(path: Path, entry: Dict[str, object], now: Optional[_dt.datetime] = None) -> InstallResult:
    """Add or update the ai-quant-lab entry in one config file."""
    data = _read_config(path)
    created = not path.exists()
    servers = dict(data.get("mcpServers", {}))
    replaced = SERVER_NAME in servers
    others = sorted(k for k in servers if k != SERVER_NAME)
    servers[SERVER_NAME] = entry
    data["mcpServers"] = servers

    backup = None
    if not created:
        stamp = (now or _dt.datetime.now()).strftime("%Y%m%d-%H%M%S")
        backup = path.with_name(f"{path.name}.bak-{stamp}")
        backup.write_bytes(path.read_bytes())

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    json.loads(tmp.read_text(encoding="utf-8"))  # never replace the real file with something unreadable
    os.replace(tmp, path)
    return InstallResult(path, created, replaced, backup, others)


def install(uvx_path: str, source: str = DEFAULT_SOURCE, env: Optional[Mapping[str, str]] = None,
            platform: Optional[str] = None, home: Optional[Path] = None) -> List[InstallResult]:
    """Register the server in every Claude Desktop config found. All files are checked before any is written."""
    paths = config_paths(env=env, platform=platform, home=home)
    for p in paths:
        _read_config(p)  # fail before writing anything if any file is unreadable
    entry = server_entry(uvx_path, source)
    return [install_into(p, entry) for p in paths]
