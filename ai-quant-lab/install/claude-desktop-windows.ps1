# ai-quant-lab installer for Claude Desktop on Windows.
#
# Run in PowerShell:
#   powershell -ExecutionPolicy ByPass -c "irm https://raw.githubusercontent.com/danieljkoo92/ECC/main/ai-quant-lab/install/claude-desktop-windows.ps1 | iex"
#
# What it does:
#   1. Installs uv (Astral's Python tool runner) if it is not already there.
#      uv fetches its own Python, so nothing else needs installing.
#   2. Downloads ai-quant-lab and runs its self-test.
#   3. Adds ai-quant-lab to Claude Desktop's MCP servers. Every existing server
#      and setting is kept, and the config file is backed up first.

$ErrorActionPreference = "Stop"
$Source = "https://github.com/danieljkoo92/ECC/archive/refs/heads/main.zip#subdirectory=ai-quant-lab"

function Find-Uvx {
    $candidates = @(
        (Join-Path $env:USERPROFILE ".local\bin\uvx.exe"),
        (Join-Path $env:LOCALAPPDATA "Microsoft\WinGet\Links\uvx.exe")
    )
    foreach ($c in $candidates) { if (Test-Path $c) { return $c } }
    $cmd = Get-Command uvx -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    return $null
}

Write-Host ""
Write-Host "ai-quant-lab -> Claude Desktop" -ForegroundColor Cyan

$uvx = Find-Uvx
if (-not $uvx) {
    Write-Host "Step 1 of 3: installing uv (one time only)..."
    # A separate PowerShell process, so the Astral installer cannot end this one.
    powershell -NoProfile -ExecutionPolicy ByPass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    $uvx = Find-Uvx
    if (-not $uvx) {
        Write-Host "  Trying winget instead..."
        winget install --id astral-sh.uv -e --accept-source-agreements --accept-package-agreements
        $env:Path = [Environment]::GetEnvironmentVariable("Path", "User") + ";" + [Environment]::GetEnvironmentVariable("Path", "Machine")
        $uvx = Find-Uvx
    }
    if (-not $uvx) { throw "uv could not be installed. Nothing in Claude was changed." }
} else {
    Write-Host "Step 1 of 3: uv is already installed."
}

Write-Host "Step 2 of 3: downloading ai-quant-lab and checking it works."
Write-Host "            The first time takes a few minutes."
& $uvx --from $Source quantlab-mcp --selftest
if ($LASTEXITCODE -ne 0) { throw "The self-test failed. Nothing in Claude was changed." }

Write-Host "Step 3 of 3: adding ai-quant-lab to Claude Desktop..."
& $uvx --from $Source quantlab-mcp --install-claude-desktop --uvx-path $uvx
if ($LASTEXITCODE -ne 0) { throw "The Claude config could not be updated. Nothing was changed." }

Write-Host ""
Write-Host "Done." -ForegroundColor Green
Write-Host "1. Fully quit Claude: right-click the Claude icon near the clock (bottom-right) and choose Quit."
Write-Host "2. Open Claude again."
Write-Host "3. Ask: Test NBA betting systems with ai-quant-lab"
