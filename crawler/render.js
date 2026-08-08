#!/usr/bin/env node
// Atlas render helper: load a Freenet gateway "shell" URL in headless Chromium,
// wait for the sandboxed WASM/SPA frame to populate, and emit JSON describing it
// so the Rust crawler can extract outbound links and a description from a page
// that renders client-side (a static fetch would see only the loader).
//
// Usage:  node render.js <shellUrl> [--shot <pngPath>]
// Output (stdout, one JSON object):
//   { "ok": true, "status": <httpStatus>, "url": <frameUrl>, "html": ..., "text": ... }
//   { "ok": true, "status": ..., "pages": [ { hash, html, text }, ... ] }  (--enumerate)
//   ...plus "partial": true when a watchdog timeout cut enumeration short.
// Every page carries BOTH: `html` is mined for links, and `text` is the content
// region, which the caller needs to DESCRIBE a site. An app-hosted site is one
// locator with several pages, so describing it from the entry page alone judged a
// site on whatever its landing route happened to be — usually the app's default
// stub. Do not drop `text` to save output bytes: it is the content region, far
// smaller than the `html` on the same page, and removing it silently restores that
// bug (a site whose content is one page over goes back to reading as too thin).
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
// The caller's acceptance threshold, mirrored so the stabilization predicate below
// can agree with it. THE RUST CONSTANT IS AUTHORITATIVE: `MIN_DESCRIBABLE_CHARS` in
// crawler/src/main.rs is what actually decides whether a rendered page is indexed or
// deferred as too thin; this is a copy, and the two MUST be changed together.
//
// They disagreed, and that was the bug. The loop below declared a page "stable" from
// the FRAME BODY's text length, while the caller judged the same page on its CONTENT
// REGION — so a Freenet app that had painted its shell but not yet received node data
// satisfied the stability test (chrome alone clears the 40-character floor) and was
// handed back at ~7s with its content region still a skeleton. Measured: 4 of 32
// locators in one crawl, e.g. BaroShare captured at 87 characters ("Connecting to
// your Freenet node...") on one run and 597 on the next, while every run exited at
// 7-9s and left ~17s of the poll budget unspent.
//
// Set `ATLAS_MIN_DESCRIBABLE_CHARS` to have the Rust side pass its own value and
// retire the copy; until then, a change on either side without the other silently
// reopens the disagreement. There is no JS test harness here, so the drift is not
// caught mechanically — main.rs's source-pin tests over this file are the place to
// add that check.
const MIN_DESCRIBABLE_CHARS = (() => {
  const n = Number.parseInt(process.env.ATLAS_MIN_DESCRIBABLE_CHARS || '', 10);
  return Number.isFinite(n) && n >= 0 ? n : 200;
})();
const POLL_STEP_MS = 1000;
const HASH_SETTLE_MS = 2500; // re-render after a hashchange (no reload, no reconnect)
const ENUM_RESERVE_MS = 8000;  // leave room to close the browser and emit
const WATCHDOG_MS = 55000; // absolute backstop so we never hang the crawler
// Process start, so every deadline below is measured against the SAME origin the
// watchdog uses. A deadline computed later in the run silently grants itself a full
// fresh budget from that point.
const startedAt = Date.now();

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

// When true, refuse to fall back to the whole body if no content region has
// substance.
//
// The fallback is correct for an ordinary web page, which need not mark up a content
// region — but it is exactly wrong for an APP shell, where the body IS the chrome.
// That is how a site whose `<main>` had not rendered yet still got described from
// Delta's sidebar (which lists every visited site by name), producing a title lifted
// from an unrelated site. With this set, such a page yields no text at all and the
// caller's minimum-content guard defers it for free instead of describing chrome.
const requireContent = args.includes('--require-content');

// Runs IN THE PAGE. Single implementation on purpose: both the text we hand back and
// the stabilization predicate in the poll loop go through it, so the "is this page
// done?" question and the "is this page describable?" question can never be answered
// off different measurements again. Inlining a second copy at either call site is how
// the two drifted the first time.
//
// `countOnly` returns the size instead of the text: the poll asks this once a second
// and only needs to compare against a threshold, and shipping a whole content region
// across the CDP boundary every poll is wasted bandwidth on a page that may carry
// megabytes of it.
function contentRegionProbe({ sels, strict, countOnly }) {
  let text = '';
  for (const s of sels) {
    const el = document.querySelector(s);
    const t = el && el.innerText ? el.innerText.trim() : '';
    if (t.length > 80) {
      text = t;
      break;
    }
  }
  if (!text) text = strict ? '' : document.body ? document.body.innerText : '';
  // Unicode SCALARS, not UTF-16 code units, because the Rust gate counts `chars()`:
  // an emoji is 2 here and 1 there. Measuring the permissive way would let this loop
  // call a page describable that the caller then rejects, which is precisely the
  // disagreement that sharing one function exists to prevent.
  return countOnly ? [...text.trim()].length : text;
}

async function contentText(frame) {
  try {
    return await frame.evaluate(contentRegionProbe, {
      sels: CONTENT_SELECTORS,
      strict: requireContent,
      countOnly: false,
    });
  } catch (_) {
    return '';
  }
}

// Size of the same region, for the poll loop. Deliberately NOT a wrapper around the
// function above: a source pin in crawler/src/main.rs counts that function's call
// sites to catch an enumerated page losing its text capture, and a fourth call would
// trip it. Both go through `contentRegionProbe`, which is the part that must not fork.
async function contentRegionChars(frame) {
  try {
    const n = await frame.evaluate(contentRegionProbe, {
      sels: CONTENT_SELECTORS,
      strict: requireContent,
      countOnly: true,
    });
    return Number.isFinite(n) ? n : 0;
  } catch (_) {
    return 0;
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

    // Count of WebSockets the page currently holds open, which is how we tell "thin
    // because it is still waiting on the node" from "thin because that is the whole
    // page". A client-side Freenet app reaches the node over a WebSocket, so a page
    // holding one open with a still-empty content region has an outstanding answer
    // coming; a page holding none does not, and no amount of extra waiting will
    // change it.
    //
    // This is what keeps the longer wait CONDITIONAL. Most thin locators are thin
    // permanently — a single-image page whose entire text is "Served from Freenet /
    // <W>x<H> / <size> / Copy link", or a bare form — and probes held those flat from
    // t=1s to t=90s. Neither opens a WebSocket, so both still settle on the old
    // ~7s path instead of each burning the full 25s poll budget for nothing.
    // Registered on the page, not a frame: the app runs in the gateway's __sandbox
    // iframe and its socket surfaces here (verified against Delta and BaroShare).
    let liveSockets = 0;
    page.on('websocket', (ws) => {
      liveSockets++;
      // Decrement once even if both events fire, or a socket that errors then closes
      // drives the count negative and permanently disables the wait for this page.
      let counted = true;
      const gone = () => {
        if (counted) {
          counted = false;
          liveSockets--;
        }
      };
      ws.on('close', gone);
      ws.on('socketerror', gone);
    });

    const resp = await page
      .goto(url, { waitUntil: 'domcontentloaded', timeout: NAV_TIMEOUT_MS })
      .catch(() => null);
    const status = resp ? resp.status() : 0;

    // Wait for the sandbox app to finish rendering. Many Freenet apps (e.g.
    // Delta) paint their chrome within ~1s but only fetch and render their real
    // content (including outbound links) seconds later, so we cannot stop at the
    // first sign of text. Instead we poll until content STABILIZES: text length,
    // anchor count and CONTENT-REGION size stop growing for two consecutive polls,
    // after a minimum settle time, bounded by an absolute timeout.
    //
    // The content-region size is in that list because the frame body alone answered
    // the wrong question. An app shell's chrome clears the 40-character floor on its
    // own and then sits flat while the real content is still in flight, so the loop
    // called the page settled and the caller — which measures only the content region
    // — rejected it as too thin. Same page, two measurements, and the renderer was
    // reporting on the one that could not see the thing being waited for.
    const deadline = Date.now() + POLL_TIMEOUT_MS;
    const minSettle = Date.now() + MIN_SETTLE_MS;
    let frame = null;
    let prevLen = -1;
    let prevA = -1;
    let prevChars = -1;
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
      const chars = await contentRegionChars(frame);
      const grew = len !== prevLen || a !== prevA || chars !== prevChars;
      const hasContent = len > 40 || a > 0;
      // Exactly the caller's acceptance test. Below it, the render is going to be
      // thrown away, so "stable" is not a useful thing to be.
      const describable = chars >= MIN_DESCRIBABLE_CHARS;
      // Not settled, however flat it looks: the content region is still too thin to
      // index AND the page is holding a node connection open, so the data that fills
      // it has not arrived yet. Flatness is what this state LOOKS like from the DOM —
      // a shell finished painting and then waits — which is why flatness alone was
      // never enough to stop on. Keep polling to the deadline; a page with no node
      // connection, or one already describable, is unaffected and still stops early.
      const awaitingNode = !describable && liveSockets > 0;
      if (!grew && hasContent && !awaitingNode && Date.now() > minSettle) {
        if (++stable >= 2) break;
      } else {
        stable = 0;
      }
      prevLen = len;
      prevA = a;
      prevChars = chars;
      // Budget genuinely exhausted: take the thin result rather than return nothing.
      // The caller defers a too-thin page for free and re-renders it on the next
      // pass, so a slow node costs a round trip, not an entry.
      if (Date.now() > deadline) break;
      await page.waitForTimeout(POLL_STEP_MS);
    }

    frame = await pickFrame(page);
    const html = await frame.evaluate(() => document.documentElement.outerHTML).catch(() => '');
    const text = await contentText(frame);

    if (shotPath) {
      // Screenshot the VIEWPORT only (fullPage: false), not the whole scrollable
      // page: this is deliberately what the visitor sees on arrival without
      // scrolling, because that is the exact thing Atlas's `landing` field is
      // defined as ("what a visitor sees IMMEDIATELY on following this
      // locator" -- see Classification::landing). A full-page capture would
      // measure something the classifier is not supposed to be judging.
      //
      // JPEG, not PNG: OpenAI's vision pricing is resolution-based (tile count),
      // not file-size based -- a fixed 1200x900 viewport costs the same input
      // tokens regardless of format or quality -- so there is nothing to lose by
      // compressing. Measured on real pages: PNG ~157 KB, JPEG q70 ~33-90 KB.
      // Roughly a third to half the base64 payload per call for identical
      // classification cost, which matters at the volume of a periodic
      // re-verification sweep.
      await page
        .screenshot({ path: shotPath, fullPage: false, type: 'jpeg', quality: 70 })
        .catch(() => {});
    }

    // --enumerate: having settled page 1, walk the rest by hash. Each step is a
    // hashchange, not a navigation, so the app keeps its connection and state.
    if (enumResource && enumMax > 0) {
      const entryHash = await frame.evaluate(() => location.hash).catch(() => '');
      const pages = [{ hash: entryHash, html, text }];
      const seenHashes = new Set();
      let entryRevisits = 0;
      let truncated = false;
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
      // Measured from process START, not from here. Computing it after the first
      // load granted a fresh full reserve from whatever time it already was, so a
      // 30s load put the loop's own deadline past the watchdog and ENUM_RESERVE_MS
      // stopped bounding anything — the watchdog became the real terminator, which
      // is the path that emits nothing but a timeout.
      const stopBy = startedAt + WATCHDOG_MS - ENUM_RESERVE_MS;
      for (let i = 1; i <= enumMax; i++) {
        if (Date.now() > stopBy) { truncated = true; break; }
        const f = await pickFrame(page);
        try {
          await f.evaluate((h) => { window.location.hash = h; }, `#${enumResource}/${i}`);
        } catch (_) { truncated = true; break; }
        await page.waitForTimeout(HASH_SETTLE_MS);
        const f2 = await pickFrame(page);
        const got = await f2
          .evaluate(() => ({
            hash: location.hash,
            html: document.documentElement.outerHTML,
          }))
          .catch(() => null);
        if (!got) { truncated = true; break; }
        // Content-region text per page, extracted the same way as the entry page.
        // The caller needs this to DESCRIBE a multi-page site: stripping the raw
        // HTML instead would pull in the app's chrome, which is the contamination
        // `--require-content` exists to prevent.
        got.text = await contentText(f2);
        // Walking back onto the ENTRY page is not "out of pages". The entry is
        // whatever the sources file named, so a walk that starts at #res/3 passes
        // through its own entry on step 3 — breaking there lost pages 4..N for
        // every locator that named anything but page 1. Skip it instead, since it
        // is already pages[0]. Only ONCE, though: an app that ignores hash changes
        // entirely lands here every step, and it should stop rather than spend the
        // settle delay on each remaining one.
        if (got.hash === entryHash) {
          if (entryRevisits++) break;
          continue;
        }
        // The app ran out of pages: it re-served one we already captured on THIS
        // walk. Without this, a 3-page site spent most of the remaining watchdog
        // re-capturing identical DOMs and pushing them toward the output cap, then
        // fed the same HTML to link extraction nine times. Note this does not fire
        // for an app that answers an out-of-range page with a fresh empty page
        // under a fresh hash, so the page cap is the real bound there.
        if (seenHashes.has(got.hash)) break;
        seenHashes.add(got.hash);
        pages.push(got);
      }
      await browser.close();
      clearTimeout(watchdog);
      // `partial` marks a walk that STOPPED EARLY rather than running out of pages.
      // It used to be set only on the hard-watchdog path, so a wall-clock or
      // evaluate-failure truncation was indistinguishable from a complete walk —
      // and the caller then made a permanent indexing and safety decision from
      // whatever prefix of the site it happened to get before the clock ran out.
      emit({ ok: true, status, url: frame.url(), html, text, pages, partial: truncated });
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
