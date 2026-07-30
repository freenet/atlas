#!/usr/bin/env node
// Atlas render helper: load a Freenet gateway "shell" URL in headless Chromium,
// wait for the sandboxed WASM/SPA frame to populate, and emit JSON describing it
// so the Rust crawler can extract outbound links and a description from a page
// that renders client-side (a static fetch would see only the loader).
//
// Usage:  node render.js <shellUrl> [--shot <pngPath>]
// Output (stdout, one JSON object):
//   { "ok": true, "status": <httpStatus>, "url": <frameUrl>, "html": ..., "text": ... }
//   { "ok": true, "status": ..., "pages": [ { hash, html }, ... ] }   (--enumerate)
//   ...plus "partial": true when a watchdog timeout cut enumeration short.
// Extra pages carry no `text`: the caller mines them for LINKS only, and a full
// innerText per page doubled the output against RENDER_MAX_BYTES for nothing.
//   { "ok": false, "error": <message> }
//
// `status` is the top-level HTTP status. The caller MUST reject a non-2xx render:
// the gateway answers an absent contract with a 500 whose body ("Contract not
// cached yet: <id>") renders as perfectly ordinary text, so "the frame is not
// empty" does not mean "this is a page worth indexing".
//
// The page content is UNTRUSTED. This helper only observes the DOM; it never
// executes page-provided instructions and the caller treats text as data.

const playwright = require('playwright');

const args = process.argv.slice(2);
const url = args[0];
const shotIdx = args.indexOf('--shot');
const shotPath = shotIdx >= 0 ? args[shotIdx + 1] : null;
// --enumerate <resource> <maxPages>: walk an app-hosted resource's pages in ONE
// browser session by driving location.hash, instead of paying a fresh navigation
// (and a fresh WebSocket connect + contract fetch) per page. Measured on Delta:
// ~14s for the first load, then ~2.5s per additional page.
//
// Needed because some apps render their internal navigation as click handlers
// rather than <a href> elements, so a link-following crawler cannot reach any page
// but the one it was pointed at. Delta does exactly that: its page list is
// clickable divs, so every page except the entry point was invisible.
const enumIdx = args.indexOf('--enumerate');
const enumResource = enumIdx >= 0 ? args[enumIdx + 1] : null;
const enumMaxRaw = enumIdx >= 0 ? Number.parseInt(args[enumIdx + 2], 10) : 0;
// `|| 12` would turn an explicit 0 (enumeration off) back into the default, because
// 0 is falsy. Check for NaN instead.
const enumMax = Number.isFinite(enumMaxRaw) ? Math.min(Math.max(enumMaxRaw, 0), 40) : 12;

const NAV_TIMEOUT_MS = 30000;
const POLL_TIMEOUT_MS = 25000; // max time to wait for the WASM frame to populate
const MIN_SETTLE_MS = 6000; // always let content load this long before accepting "stable"
const POLL_STEP_MS = 1000;
const HASH_SETTLE_MS = 2500; // re-render after a hashchange (no reload, no reconnect)
const ENUM_RESERVE_MS = 8000;  // leave room to close the browser and emit
const WATCHDOG_MS = 55000; // absolute backstop so we never hang the crawler

// Write the JSON result and exit only once stdout has fully drained. Calling
// process.exit() immediately after a large write truncates output at the OS
// pipe buffer (~64KB), so we wait for the write callback before exiting.
function emit(obj) {
  process.stdout.write(JSON.stringify(obj), () => process.exit(0));
}

// Pages captured so far, at module scope so the watchdog can return them.
//
// Losing them was a real failure mode: enumeration costs ~2.7s per page, so a
// 12-page walk sits close to the watchdog, and a slow first load tipped it over —
// at which point the OLD watchdog discarded every page INCLUDING the entry page
// that had already been captured. The crawl then fell back to a static fetch, saw
// the bare loader, and captured nothing, re-burning the full watchdog every hub
// interval without ever self-healing. Returning partial results makes a timeout
// degrade instead of fail.
let captured = null;

// Absolute backstop: if anything wedges, exit with whatever we have rather than
// hang or throw it away.
const watchdog = setTimeout(() => {
  if (captured && captured.pages && captured.pages.length) {
    emit({ ...captured, ok: true, partial: true });
  } else {
    emit({ ok: false, error: 'render watchdog timeout' });
  }
}, WATCHDOG_MS);
watchdog.unref();

// Text used for DESCRIPTION and SAFETY RATING, taken from the page's content region
// rather than the whole frame.
//
// This mattered a great deal. An app shell's chrome is in the frame text too, and
// Delta's sidebar lists every site the node has ever visited BY NAME — so describing
// from the frame text fed the LLM a menu of other sites' names, and it picked one as
// the title. The result was 16 live index entries with cross-contaminated titles
// ("Run Freenet in Docker" eight times, for eight unrelated sites). The site content
// was being fetched correctly all along; only the text handed to the LLM was wrong.
//
// The preference list is generic HTML semantics, not app-specific: a page that marks
// up its content region gets described from it, and anything else falls back to the
// full body exactly as before.
const CONTENT_SELECTORS = ['main', '[role=main]', 'article', '#content', '.content'];

async function contentText(frame) {
  try {
    return await frame.evaluate((sels) => {
      for (const s of sels) {
        const el = document.querySelector(s);
        const t = el && el.innerText ? el.innerText.trim() : '';
        // Require some substance: an empty or near-empty <main> means the app has
        // not rendered into it yet, and the body text is the better signal.
        if (t.length > 80) return t;
      }
      return document.body ? document.body.innerText : '';
    }, CONTENT_SELECTORS);
  } catch (_) {
    return '';
  }
}

// Pick the frame most likely to hold the rendered app: prefer one whose URL
// carries the gateway sandbox marker, else the frame with the most text.
async function pickFrame(page) {
  const frames = page.frames();
  const sandbox = frames.filter((f) => f.url().includes('__sandbox'));
  const candidates = sandbox.length ? sandbox : frames;
  let best = null;
  let bestLen = -1;
  for (const f of candidates) {
    let len = 0;
    try {
      len = await f.evaluate(() => (document.body ? document.body.innerText.length : 0));
    } catch (_) {}
    if (len > bestLen) {
      bestLen = len;
      best = f;
    }
  }
  return best || page.mainFrame();
}

(async () => {
  if (!url) {
    emit({ ok: false, error: 'no url' });
    return;
  }
  let browser;
  try {
    browser = await playwright.chromium.launch({ headless: true });
    const page = await browser.newPage({ viewport: { width: 1200, height: 900 } });
    const resp = await page
      .goto(url, { waitUntil: 'domcontentloaded', timeout: NAV_TIMEOUT_MS })
      .catch(() => null);
    const status = resp ? resp.status() : 0;

    // Wait for the sandbox app to finish rendering. Many Freenet apps (e.g.
    // Delta) paint their chrome within ~1s but only fetch and render their real
    // content (including outbound links) seconds later, so we cannot stop at the
    // first sign of text. Instead we poll until content STABILIZES: text length
    // and anchor count stop growing for two consecutive polls, after a minimum
    // settle time, bounded by an absolute timeout.
    const deadline = Date.now() + POLL_TIMEOUT_MS;
    const minSettle = Date.now() + MIN_SETTLE_MS;
    let frame = null;
    let prevLen = -1;
    let prevA = -1;
    let stable = 0;
    for (;;) {
      frame = await pickFrame(page);
      let len = 0;
      let a = 0;
      try {
        [len, a] = await frame.evaluate(() => [
          document.body ? document.body.innerText.trim().length : 0,
          document.querySelectorAll('a').length,
        ]);
      } catch (_) {}
      const grew = len !== prevLen || a !== prevA;
      const hasContent = len > 40 || a > 0;
      if (!grew && hasContent && Date.now() > minSettle) {
        if (++stable >= 2) break;
      } else {
        stable = 0;
      }
      prevLen = len;
      prevA = a;
      if (Date.now() > deadline) break;
      await page.waitForTimeout(POLL_STEP_MS);
    }

    frame = await pickFrame(page);
    const html = await frame.evaluate(() => document.documentElement.outerHTML).catch(() => '');
    const text = await contentText(frame);

    if (shotPath) {
      // Screenshot the whole page viewport: the gateway shell is a thin wrapper
      // whose iframe fills the viewport, so this captures the rendered site.
      await page.screenshot({ path: shotPath, fullPage: false }).catch(() => {});
    }

    // --enumerate: having settled page 1, walk the rest by hash. Each step is a
    // hashchange, not a navigation, so the app keeps its connection and state.
    if (enumResource && enumMax > 0) {
      const entryHash = await frame.evaluate(() => location.hash).catch(() => '');
      const pages = [{ hash: entryHash, html }];
      const seenHashes = new Set([entryHash]);
      captured = { status, url: frame.url(), html, text, pages };
      // From 1, not 2: the entry URL is whatever the sources file named, which need
      // not be page 1 — starting at 2 meant page 1 was never enumerated (and the
      // entry page was captured twice) whenever the hub named any other page. The
      // hash set makes revisiting the entry page a no-op instead of a duplicate.
      //
      // Bounded by remaining WALL CLOCK as well as page count, because the page
      // count alone does not bound time: a 12-page walk is ~30s of hashing on top
      // of a first load that can itself take 30s, and overshooting the watchdog
      // used to lose everything.
      const stopBy = Date.now() + WATCHDOG_MS - ENUM_RESERVE_MS;
      for (let i = 1; i <= enumMax; i++) {
        if (Date.now() > stopBy) break;
        const f = await pickFrame(page);
        try {
          await f.evaluate((h) => { window.location.hash = h; }, `#${enumResource}/${i}`);
        } catch (_) { break; }
        await page.waitForTimeout(HASH_SETTLE_MS);
        const f2 = await pickFrame(page);
        const got = await f2
          .evaluate(() => ({
            hash: location.hash,
            html: document.documentElement.outerHTML,
          }))
          .catch(() => null);
        if (!got) break;
        // The app ran out of pages: it either stopped moving or re-served a page we
        // already have. Without this, a 3-page site spent most of the remaining
        // watchdog re-capturing identical DOMs and pushing them toward the output
        // cap, then fed the same HTML to link extraction nine times.
        if (seenHashes.has(got.hash)) break;
        seenHashes.add(got.hash);
        pages.push(got);
      }
      await browser.close();
      clearTimeout(watchdog);
      emit({ ok: true, status, url: frame.url(), html, text, pages });
      return;
    }

    await browser.close();
    clearTimeout(watchdog);
    emit({ ok: true, status, url: frame.url(), html, text });
  } catch (e) {
    try { if (browser) await browser.close(); } catch (_) {}
    clearTimeout(watchdog);
    emit({ ok: false, error: String(e && e.message ? e.message : e) });
  }
})();
