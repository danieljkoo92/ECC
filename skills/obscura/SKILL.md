---
name: obscura
description: Use Obscura, the open-source Rust headless browser, for JavaScript page loading, web extraction, screenshots, PDF export, and CDP automation with Puppeteer or Playwright. Use when a task needs a lightweight browser for scraping, site audits, lead research, competitor monitoring, or AI-agent browsing without launching Chromium.
origin: community (h4ckf0r0day/obscura, Apache-2.0)
---

# Obscura

Obscura is a headless browser engine written in Rust. It embeds V8, runs real
JavaScript, speaks the Chrome DevTools Protocol (CDP), and works as a drop-in
replacement for headless Chrome with Puppeteer and Playwright. Source and
releases: `github.com/h4ckf0r0day/obscura` (Apache-2.0). This skill is adapted
from the upstream `skills/obscura/SKILL.md`.

## When to Use

- Scraping or extracting text and data from JavaScript-heavy pages.
- Auditing a prospect or client website: screenshots, page text, titles, links.
- Lead research from public directories and listings.
- Competitor and price monitoring across many URLs.
- Giving an AI agent a cheap browser (MCP or CDP) when Chromium is too heavy.

Do not use it to bypass logins, paywalls, CAPTCHAs, or a site's terms of
service. Prefer official APIs when a site offers one.

## How It Works

### Install

Download a release binary (Linux, macOS, Windows builds are published):

```bash
curl -LO https://github.com/h4ckf0r0day/obscura/releases/latest/download/obscura-x86_64-linux.tar.gz
tar xzf obscura-x86_64-linux.tar.gz
./obscura fetch https://example.com --eval "document.title"
```

Release archives include rendering (screenshots, PDF). From source:

```bash
cargo build --release -p obscura-cli --bins --features render
```

### Modes

| Mode | Command | Use for |
|------|---------|---------|
| CLI fetch | `obscura fetch <url>` | One page: text, eval, screenshot |
| CLI scrape | `obscura scrape <urls...>` | Many URLs, no per-URL screenshot |
| CDP server | `obscura serve --port 9222` | Puppeteer / Playwright scripts |
| MCP | `obscura mcp` or `obscura mcp --http --port 3000` | Agent browsing tools |

### Key behavior

- An omitted `--wait` uses adaptive settling capped at five seconds;
  `--wait N` is a fixed delay in seconds. `--timeout` bounds navigation.
- `--dump original` returns the raw HTTP response without DOM processing.
- Local addresses (`127.0.0.1`, RFC1918) need the global
  `--allow-private-network` flag.
- MCP: navigate first, then refresh the snapshot after every click, scroll, or
  rerender, because element references change.
- Render-enabled MCP builds expose `browser_screenshot` (PNG) and
  `browser_pdf`. Screencast frames are CDP-only; each
  `Page.screencastFrame` must be acknowledged with `Page.screencastFrameAck`.
- PDF output is raster-backed: no selectable text, outlines, or headers.

### Limits

Obscura is not Chrome. Long-tail CSS, service workers, some Web APIs, native
media, GPU effects, and font rasterization can differ. When output looks
wrong, confirm navigation succeeded and the capture is nonblank before
debugging, and fall back to Playwright with Chromium for pixel-exact work.

## Examples

Page title and visible text:

```bash
obscura fetch https://example.com --eval "document.title"
obscura fetch https://example.com --dump text
```

Full-page audit screenshot, then the bottom of the page:

```bash
obscura fetch https://example.com --screenshot top.png
obscura fetch https://example.com \
  --eval "window.scrollTo(0, document.documentElement.scrollHeight)" \
  --screenshot bottom.png
```

Playwright over CDP:

```js
const { chromium } = require('playwright');

(async () => {
  const browser = await chromium.connectOverCDP('http://127.0.0.1:9222');
  const page = await browser.newPage();
  await page.goto('https://example.com');
  console.log(await page.innerText('body'));
  await page.screenshot({ path: 'page.png', fullPage: true });
  await browser.close();
})();
```

Agent browsing through MCP (Claude Code):

```bash
claude mcp add obscura -- obscura mcp
```
