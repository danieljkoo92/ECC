---
name: build-your-own-x
description: Reference map of ~390 from-scratch tutorials (codecrafters-io/build-your-own-x) covering databases, web servers, search engines, bots, blockchains, AI models, neural networks, games, browsers, Git, Docker, shells, programming languages and more. Use whenever planning, designing, explaining or building any system or app component (e.g. "how would a queue/database/search/bot/order system work", architecture for Orderflow or Perfect Day, choosing what to build vs buy), or when Daniel asks how a technology works under the hood.
---

# Build Your Own X — reference map

Source: github.com/codecrafters-io/build-your-own-x. Full tutorial list with links is in `references/catalog.md`, grouped by `## Build your own <Thing>`.

## When to use
- Planning or building any app piece (storage, search, messaging, bots, payments ledger, web server, UI framework).
- Explaining how a technology works, in plain words.
- Deciding build-vs-buy: if a tutorial exists, the piece is well understood; still prefer a proven service for anything shipping to customers.

## How it works
1. Match the task to one or more categories in `references/catalog.md` (grep the `##` headings).
2. Pick 1–3 tutorials in the stack being used (JavaScript/TypeScript/Python first).
3. Use them as design background. Claude does the building. Never hand Daniel code or ask him to follow a tutorial himself.
4. When explaining, give steps only, no theory, and mention at most one tutorial link.

## Examples
- "How should Orderflow store orders?" → check *Database* and *Web Server*, then recommend a hosted database and explain the design in 3–5 steps.
- "Could Perfect Day have a chat assistant?" → check *Bot* and *AI Model*.
