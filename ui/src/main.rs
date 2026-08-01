#![allow(non_snake_case)]
//! Atlas Discover: a read-only front door to Freenet. Connects to the local node
//! over the WebSocket command API, GET+SUBSCRIBEs the index contract, and renders
//! browse + client-side search + Open. No identity, no writes, no delegate.

#[cfg(target_arch = "wasm32")]
use std::cell::RefCell;

#[cfg(test)]
use atlas_common::SubjectId;
use atlas_common::{IndexEntry, IndexState, Kind, Locator};
use dioxus::prelude::*;
#[cfg(target_arch = "wasm32")]
use freenet_stdlib::client_api::{
    ClientRequest, ContractRequest, ContractResponse, Error, HostResponse, WebApi,
};
#[cfg(target_arch = "wasm32")]
use freenet_stdlib::prelude::ContractInstanceId;
// Not gated to wasm32: used by `set_shell_title`, which compiles on native too.
use wasm_bindgen::JsValue;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_futures::spawn_local;

/// The index contract instance id, baked in at build time. REQUIRED: there is
/// deliberately no default.
///
/// A default is a trap here. The index re-keys on any `common/` change, and
/// `atlasctl` derives its target from the committed WASM, so after a re-key the
/// curator writes to the new address while a UI built with a stale default keeps
/// reading the old one. The failure is silent: the site renders the pre-re-key
/// snapshot indefinitely and simply stops showing anything new, with nothing
/// erroring.
///
/// (The published UI has NOT actually regressed this way — the 0.8.3 publish did
/// pass the migrated id via this env var. An earlier version of this comment
/// asserted it had, inferring a live bug from the stale default without checking the
/// deployed bytes. The trap was real; the regression was not.)
///
/// `env!` turns forgetting it into a build failure instead. Get the value from
/// `atlasctl key`.
const INDEX_ID: &str = env!(
    "ATLAS_INDEX_ID",
    "ATLAS_INDEX_ID must be set when building the Atlas UI — run `atlasctl key` \
     to get the current index contract id. A stale or defaulted value silently \
     freezes the published site at an old index generation."
);

/// `env!` is satisfied by an EMPTY value, which is exactly what
/// `ATLAS_INDEX_ID=$(atlasctl key)` produces when `atlasctl` fails (command
/// substitution discards the exit status). The build would succeed and the site
/// would show "bad index id" forever, which is the same silent-failure shape this
/// constant exists to prevent. Check the shape at compile time.
/// Range, not exactly 43-or-44: a uniformly random 32-byte value base58-encodes to
/// 44 chars ~94% of the time and 43 chars ~5.7%, but ~0.1% of legitimate ids are
/// SHORTER, and hard-failing the build on one of those would blame the operator for
/// a valid id. The point is to catch empty or obviously-truncated values.
const _: () = assert!(
    INDEX_ID.len() >= 40 && INDEX_ID.len() <= 44,
    "ATLAS_INDEX_ID must be a base58 contract instance id (40-44 chars); an empty value usually means the command substitution supplying it failed"
);

// `allow(dead_code)` below is scoped per-item, not crate-wide, and only for
// `not(target_arch = "wasm32")`. These items are unreferenced from native's
// stub `main` (see below) but exist so `cargo test -p atlas-ui` can exercise
// the pure logic that references them. On wasm32 the real `main` uses all of
// them, so the allow evaluates to nothing there — it must stay that way,
// since the wasm32 dead_code lint (promoted to an error by CI's `-D
// warnings`) is what catches an item accidentally left unreachable by a
// mis-gated `#[cfg]` on the real app. Do not widen this to crate scope or
// flip its polarity.
#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
static STATE: GlobalSignal<Option<IndexState>> = Signal::global(|| None);
#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
static STATUS: GlobalSignal<String> = Signal::global(|| "connecting…".to_string());

#[cfg(target_arch = "wasm32")]
thread_local! {
    static API: RefCell<Option<WebApi>> = const { RefCell::new(None) };
}

#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
const CSS: &str = r#"
:root { --bg:#fff; --fg:#16181d; --dim:#6b7280; --faint:#9aa1ab;
  --line:#e7e9ee; --card:#fff; --card-hover:#fafafb; --feat:#f7f7f9; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#0d0f13; --fg:#e9eaed; --dim:#9aa1ab; --faint:#6b7280;
    --line:#23272f; --card:#14171d; --card-hover:#181c23; --feat:#171b22; }
}
* { box-sizing:border-box; }
html { -webkit-text-size-adjust:100%; }
body { margin:0; background:var(--bg); color:var(--fg); line-height:1.5;
  -webkit-font-smoothing:antialiased;
  font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',system-ui,sans-serif; }
.wrap { max-width:980px; margin:0 auto; padding:2.5rem 1.25rem 4rem; }
.head { display:flex; align-items:baseline; justify-content:space-between; gap:1rem;
  padding-bottom:1.1rem; border-bottom:1px solid var(--line); margin-bottom:1.5rem; }
.brand h1 { margin:0; font-size:1.7rem; font-weight:680; letter-spacing:-0.03em; }
.brand p { margin:.2rem 0 0; color:var(--dim); font-size:.9rem; }
.count { color:var(--faint); font-size:.78rem; white-space:nowrap;
  font-variant-numeric:tabular-nums; }
.search { width:100%; padding:.72rem .95rem; font-size:.98rem; color:var(--fg);
  background:var(--card); border:1px solid var(--line); border-radius:9px; outline:none;
  transition:border-color .15s; }
.search::placeholder { color:var(--faint); }
.search:focus { border-color:var(--dim); }
.status { color:var(--dim); font-size:.8rem; margin:.7rem 0 0; }
.results { color:var(--faint); font-size:.76rem; margin:.7rem 0 0;
  font-variant-numeric:tabular-nums; }
.grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(264px,1fr));
  gap:.85rem; margin-top:1.4rem; }
.card { position:relative; border:1px solid var(--line); border-radius:11px;
  padding:1.05rem 1.1rem; background:var(--card); display:flex; flex-direction:column;
  min-height:152px; transition:border-color .15s, transform .15s, background .15s; }
.card:hover { border-color:var(--dim); transform:translateY(-2px); background:var(--card-hover); }
.card.feat { background:var(--feat); }
.card-top { display:flex; justify-content:space-between; align-items:center; margin-bottom:.55rem; }
.kind { font-size:.66rem; text-transform:uppercase; letter-spacing:.09em; color:var(--faint);
  font-weight:600; }
.card-top .kind.app { color:var(--dim); text-transform:none; letter-spacing:0; margin-left:.45rem;
  margin-right:auto; }
.star { color:var(--dim); font-size:.82rem; }
.card h3 { margin:0 0 .4rem; font-size:1.04rem; font-weight:620; letter-spacing:-0.01em;
  line-height:1.3; }
.snip { color:var(--dim); font-size:.875rem; line-height:1.45; margin:0 0 .8rem;
  display:-webkit-box; -webkit-line-clamp:3; -webkit-box-orient:vertical; overflow:hidden; }
.tags { display:flex; flex-wrap:wrap; gap:.32rem; margin:0 0 .9rem; }
.t { font-size:.68rem; color:var(--dim); background:transparent; border:1px solid var(--line);
  border-radius:5px; padding:.08rem .4rem; }
.open { margin-top:auto; align-self:flex-start; font-size:.8rem; text-decoration:none;
  color:var(--fg); border:1px solid var(--line); border-radius:7px; padding:.34rem .65rem;
  transition:border-color .15s; }
.open:hover { border-color:var(--dim); }
.open.unavail { color:var(--faint); border-style:dashed; cursor:default; }
.filter-row { display:flex; align-items:center; gap:.5rem; margin-top:.6rem;
  color:var(--dim); font-size:.8rem; }
.filter-row input { accent-color:var(--fg); }
.filter-row label { cursor:pointer; }
.empty { color:var(--dim); padding:3rem 0; text-align:center; }
.foot { margin-top:2.5rem; padding-top:1.2rem; border-top:1px solid var(--line);
  color:var(--faint); font-size:.76rem; line-height:1.6; }
"#;

// This binary exists on native only so `cargo test -p atlas-ui` can run the
// pure-function tests (the test harness supplies its own `main` and never
// calls this one). Panic rather than exit cleanly: before this file grew a
// native stub, a non-wasm32 build of this crate failed to compile outright,
// and this repo has a standing rule against turning a loud build failure
// into a silent no-op (see the `ATLAS_INDEX_ID` comment above) — if `dx`'s
// platform selection ever resolves to something other than `web` for this
// crate, this should still fail loudly instead of shipping a binary that
// does nothing.
#[cfg(not(target_arch = "wasm32"))]
fn main() {
    panic!("atlas-ui is a wasm32-only web app; this native binary exists only so `cargo test -p atlas-ui` can run its pure-function tests");
}

#[cfg(target_arch = "wasm32")]
fn main() {
    std::panic::set_hook(Box::new(|info| {
        web_sys::console::error_1(&format!("atlas panic: {info}").into());
    }));
    launch(App);
}

#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
fn App() -> Element {
    #[cfg(target_arch = "wasm32")]
    use_hook(|| {
        set_shell_title("Atlas");
        connect();
    });
    let mut query = use_signal(String::new);
    let q = query().to_lowercase();
    // Session-only (a `use_signal`, not persisted) — the ask was for the
    // DEFAULT to change, not for hiding external results permanently; anyone
    // who wants them is one click away every time. `SHOW_EXTERNAL_BY_DEFAULT`
    // is a named constant, not a literal `false` here, specifically so a test
    // can pin it directly: this function's own component-level default is not
    // otherwise reachable from a test (`App` needs a live Dioxus render), and
    // a mutation flipping this literal to `true` — silently shipping with the
    // filter showing everything, defeating the entire point of the feature —
    // passed every OTHER test in this file when tried.
    let mut show_external = use_signal(|| SHOW_EXTERNAL_BY_DEFAULT);

    // Total live entries (unfiltered) for the header count, and the filtered set
    // shown in the grid.
    let total = STATE
        .read()
        .as_ref()
        .map(|s| s.live_entries().count())
        .unwrap_or(0);
    // Whether the toggle row renders at all: the WHOLE index, independent of
    // the query and of the toggle itself. Query-independent so the row cannot
    // vanish out from under a search (a query matching zero externals must
    // not also hide the only control that explains why); toggle-independent
    // so turning it on cannot make the row (and with it, the only way to turn
    // it back off) disappear.
    let total_external = STATE
        .read()
        .as_ref()
        .map(|s| {
            s.live_entries()
                .filter(|e| !passes_external_filter(e, false))
                .count()
        })
        .unwrap_or(0);
    // What the label claims: how many hidden web links match the CURRENT
    // query, not the whole index. Using the unscoped count here reads as a
    // promise the toggle does not keep — search for something no external
    // entry matches and it would still claim N are hidden, then reveal zero
    // new results when clicked.
    let external_matching_query = STATE
        .read()
        .as_ref()
        .map(|s| {
            s.live_entries()
                .filter(|e| matches_query(e, &q))
                .filter(|e| !passes_external_filter(e, false))
                .count()
        })
        .unwrap_or(0);
    let entries: Vec<IndexEntry> = match STATE.read().as_ref() {
        Some(state) => {
            let mut v: Vec<IndexEntry> = state.live_entries().cloned().collect();
            v.sort_by(|a, b| {
                b.featured
                    .cmp(&a.featured)
                    .then(b.added_at.cmp(&a.added_at))
            });
            v.into_iter()
                .filter(|e| passes_external_filter(e, show_external()))
                .filter(|e| matches_query(e, &q))
                .collect()
        }
        None => Vec::new(),
    };
    let searching = !q.is_empty();
    let shown = entries.len();

    rsx! {
        style { dangerous_inner_html: CSS }
        div { class: "wrap",
            div { class: "head",
                div { class: "brand",
                    h1 { "Atlas" }
                    p { "Discover Freenet" }
                }
                if total > 0 {
                    div { class: "count", "{total} entries" }
                }
            }
            input {
                class: "search",
                placeholder: "Search apps, sites, and more…",
                value: "{query}",
                oninput: move |e| query.set(e.value()),
            }
            // Only shown when there is something to toggle: a reader with zero
            // external entries in the index has nothing to decide. Gated on
            // `total_external`, deliberately not the query-scoped count below
            // — this row is the only way to turn the toggle back off, so it
            // must stay put regardless of what the search box currently says.
            if total_external > 0 {
                div { class: "filter-row",
                    input {
                        r#type: "checkbox",
                        id: "show-external",
                        checked: show_external(),
                        onchange: move |e| show_external.set(e.checked()),
                    }
                    label { r#for: "show-external",
                        if show_external() {
                            "Showing regular web links too"
                        } else {
                            "Freenet only — {external_matching_query} web link{plural_s(external_matching_query)} hidden"
                        }
                    }
                }
            }
            // Only surface connection status while not yet ready (connecting,
            // looking for the index, errors); hide it in the normal case.
            if STATUS.read().as_str() != "ready" {
                div { class: "status", "{STATUS}" }
            }
            if searching && STATE.read().is_some() {
                div { class: "results",
                    if shown == 1 { "1 result" } else { "{shown} results" }
                }
            }
            if entries.is_empty() {
                div { class: "empty",
                    if STATE.read().is_none() {
                        "Loading…"
                    } else if !show_external() && external_matching_query > 0 {
                        // The filter, not the search, is why the grid is
                        // empty — everything that DID match got hidden by the
                        // toggle above. Distinguished from "nothing matches"
                        // because the fix is different: click the toggle, not
                        // change the search.
                        "Nothing on Freenet matches — {external_matching_query} web \
                         link{plural_s(external_matching_query)} do, hidden above."
                    } else {
                        "Nothing matches."
                    }
                }
            } else {
                div { class: "grid",
                    for e in entries {
                        EntryCard { key: "{e.subject_id.as_str()}", entry: e }
                    }
                }
            }
            footer { class: "foot",
                "Atlas lists what it finds on Freenet. A listing is not an endorsement; open links at your own discretion."
            }
        }
    }
}

#[component]
#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
fn EntryCard(entry: IndexEntry) -> Element {
    let external = matches!(entry.locator, Locator::External { .. });
    // Resolution goes through the state's app registry, so an app-hosted entry
    // follows the app's CURRENT address. `None` means the registry does not know
    // the app (yet), which is a curator gap rather than a broken entry: render
    // the card without an Open link instead of emitting a href that 404s.
    let href = STATE
        .read()
        .as_ref()
        .and_then(|s| s.resolve_href(&entry.locator));
    // The app name is the useful label for an app-hosted resource: "delta site"
    // says far more than "site", and without it every Delta site in the index
    // looks identical to every other.
    let host_app = match &entry.locator {
        Locator::AppResource { app, .. } => STATE
            .read()
            .as_ref()
            .and_then(|s| {
                s.apps
                    .as_ref()
                    .and_then(|r| r.get(app))
                    .map(|a| a.name.clone())
            })
            .or_else(|| Some(app.clone())),
        _ => None,
    };
    let card_class = if entry.featured { "card feat" } else { "card" };
    rsx! {
        div { class: "{card_class}",
            div { class: "card-top",
                span { class: "kind", "{kind_label(entry.kind)}" }
                if let Some(app) = host_app {
                    span { class: "kind app", "on {app}" }
                }
                if entry.featured {
                    span { class: "star", "★" }
                }
            }
            h3 { "{entry.title}" }
            p { class: "snip", "{entry.snippet}" }
            if !entry.tags.is_empty() {
                div { class: "tags",
                    for t in entry.tags.iter() {
                        span { class: "t", "{t}" }
                    }
                }
            }
            match href {
                Some(h) => rsx! {
                    a {
                        class: "open",
                        href: "{h}",
                        target: if external { "_blank" } else { "_self" },
                        "Open ↗"
                    }
                },
                None => rsx! {
                    span { class: "open unavail", title: "This app is not in the registry yet",
                        "Unavailable"
                    }
                },
            }
        }
    }
}

/// Set the browser tab title. When served by the Freenet gateway the app runs in
/// a sandboxed iframe whose parent "shell" owns the tab (and defaults its title
/// to "Freenet"), so we both set our own document title and postMessage the
/// title to the parent shell via its `__freenet_shell__` bridge (same mechanism
/// River uses).
///
/// Not gated to wasm32: `web_sys`/`js_sys`/`wasm_bindgen` compile fine on the
/// host target (they emit no-op/panicking stubs off-wasm), and this function
/// is never actually called from native's stub `main` — only `connect` and
/// `request_index` genuinely need `wasm32` to compile, because they reach
/// `freenet_stdlib::client_api::WebApi`, whose native implementation takes a
/// different argument list.
#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
fn set_shell_title(title: &str) {
    let Some(window) = web_sys::window() else {
        return;
    };
    if let Some(doc) = window.document() {
        doc.set_title(title);
    }
    let msg = js_sys::Object::new();
    let _ = js_sys::Reflect::set(
        &msg,
        &JsValue::from_str("__freenet_shell__"),
        &JsValue::TRUE,
    );
    let _ = js_sys::Reflect::set(
        &msg,
        &JsValue::from_str("type"),
        &JsValue::from_str("title"),
    );
    let _ = js_sys::Reflect::set(&msg, &JsValue::from_str("title"), &JsValue::from_str(title));
    let target = window
        .parent()
        .ok()
        .flatten()
        .unwrap_or_else(|| window.clone());
    let _ = target.post_message(&JsValue::from(msg), "*");
}

#[cfg(target_arch = "wasm32")]
fn connect() {
    let url = match ws_url() {
        Some(u) => u,
        None => return,
    };
    let ws = match web_sys::WebSocket::new(&url) {
        Ok(w) => w,
        Err(_) => {
            *STATUS.write() = "websocket error".to_string();
            return;
        }
    };
    let api = WebApi::start(
        ws,
        |res| match res {
            Ok(HostResponse::ContractResponse(ContractResponse::GetResponse { state, .. })) => {
                match ciborium::de::from_reader::<IndexState, _>(state.as_ref()) {
                    Ok(st) => {
                        *STATE.write() = Some(st);
                        *STATUS.write() = "ready".to_string();
                    }
                    Err(e) => *STATUS.write() = format!("decode error: {e}"),
                }
            }
            Ok(HostResponse::ContractResponse(ContractResponse::UpdateNotification { .. })) => {
                spawn_local(request_index());
            }
            Ok(HostResponse::ContractResponse(ContractResponse::NotFound { .. })) => {
                // The index may not be hosted on a reachable peer yet (cross-node
                // propagation). Retry rather than hang on "Loading…".
                *STATUS.write() = "looking for the index…".to_string();
                gloo_timers::callback::Timeout::new(4000, || spawn_local(request_index())).forget();
            }
            Ok(_) => {}
            Err(e) => {
                *STATUS.write() = format!("error: {e}");
                gloo_timers::callback::Timeout::new(5000, || spawn_local(request_index())).forget();
            }
        },
        |_e: Error| {},
        || {
            *STATUS.write() = "connected".to_string();
            spawn_local(request_index());
        },
    );
    API.with(|a| *a.borrow_mut() = Some(api));
}

#[cfg(target_arch = "wasm32")]
async fn request_index() {
    let id = match INDEX_ID.parse::<ContractInstanceId>() {
        Ok(i) => i,
        Err(_) => {
            *STATUS.write() = "bad index id".to_string();
            return;
        }
    };
    let req = ClientRequest::ContractOp(ContractRequest::Get {
        key: id,
        return_contract_code: false,
        subscribe: true,
        blocking_subscribe: false,
    });
    let api = API.with(|a| a.borrow_mut().take());
    if let Some(mut api) = api {
        let _ = api.send(req).await;
        API.with(|a| *a.borrow_mut() = Some(api));
    }
}

// Not gated to wasm32 — see the rationale comment on `set_shell_title` above.
#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
fn ws_url() -> Option<String> {
    let win = web_sys::window()?;
    let loc = win.location();
    let proto = loc.protocol().unwrap_or_else(|_| "http:".to_string());
    let host = loc.host().unwrap_or_default();
    let ws_proto = if proto == "https:" { "wss:" } else { "ws:" };
    let mut url = format!("{ws_proto}//{host}/v1/contract/command?encodingProtocol=native");
    if let Ok(tok) = js_sys::Reflect::get(&win, &"__FREENET_AUTH_TOKEN__".into()) {
        if let Some(t) = tok.as_string() {
            if !t.is_empty() {
                url.push_str(&format!("&authToken={t}"));
            }
        }
    }
    Some(url)
}

#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
fn kind_label(kind: Kind) -> &'static str {
    match kind {
        Kind::App => "app",
        Kind::Site => "site",
        Kind::External => "web",
    }
}

/// Per user feedback (Ivvor, River Official, 2026-07-31): "I think Atlas
/// should only show Freenet links by default. The regular web links could be
/// confusing." A `Kind::External` entry is an ordinary https:// page, not
/// something the crawler found ON Freenet, and mixed into results by default
/// it reads as "Atlas found this on Freenet" when it did not.
const SHOW_EXTERNAL_BY_DEFAULT: bool = false;

/// Should `e` be shown given the current "show external (web) links" setting?
///
/// Pulled out as a plain function (no `web_sys`/Dioxus dependency) so it is
/// unit-testable on the native target — `connect` and `request_index` still
/// need `wasm32` to compile (they reach `freenet_stdlib::client_api::WebApi`,
/// whose native signature differs), but that isn't a `web_sys` constraint.
fn passes_external_filter(e: &IndexEntry, show_external: bool) -> bool {
    // Keys on `locator`, not `kind`, matching the existing check at the Open
    // link (`let external = matches!(entry.locator, Locator::External { .. })`)
    // rather than inventing a second one. They are independent fields with no
    // enforced invariant tying them together — nothing in `IndexEntry::check`,
    // `atlasctl add`, or the on-chain contract requires `kind` to agree with
    // what `locator` actually opens — and the crawler has at least one path
    // that can label a real Freenet locator "external" (an `app:` locator that
    // fails `map_locator`'s reversal falls into the scheme-blind
    // `"external"` fallback). `kind` is a DISPLAY taxonomy; whether a link
    // takes you off Freenet is `locator`'s question, and getting this wrong in
    // that direction — a "Freenet only" filter hiding real Freenet content —
    // is the one failure mode this feature must never have.
    show_external || !matches!(e.locator, Locator::External { .. })
}

fn plural_s(n: usize) -> &'static str {
    if n == 1 {
        ""
    } else {
        "s"
    }
}

#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
fn matches_query(e: &IndexEntry, q: &str) -> bool {
    if q.is_empty() {
        return true;
    }
    e.title.to_lowercase().contains(q)
        || e.snippet.to_lowercase().contains(q)
        || e.tags.iter().any(|t| t.to_lowercase().contains(q))
}

// Native-only: everything above this point that touches `web_sys` needs the
// `wasm32` target to compile at all, so these run against the pure functions
// only (`passes_external_filter`, `plural_s`, `matches_query`, `kind_label`),
// via `cargo test -p atlas-ui` on the host target. There is no test harness in
// this crate for the rendered component tree itself — see the doc comment on
// `passes_external_filter` for why the filter decision was pulled out as a
// plain function rather than tested through `EntryCard`/`App`.
#[cfg(test)]
mod tests {
    use super::*;

    /// `kind` and `locator` set to agree, matching how the crawler's normal
    /// paths pair them. Use [`entry_mismatched`] to construct the disagreeing
    /// case the filter has to get right anyway.
    fn entry(kind: Kind) -> IndexEntry {
        let locator = match kind {
            Kind::External => Locator::External {
                url: "https://example.com".to_string(),
            },
            _ => Locator::Freenet {
                contract_id: "EqJ5YpEEV3XLqEvKWLQHFhGAac2qXzSUoE6k2zbdnXBr".to_string(),
                path: "/".to_string(),
            },
        };
        entry_with(kind, locator)
    }

    fn entry_with(kind: Kind, locator: Locator) -> IndexEntry {
        IndexEntry {
            subject_id: SubjectId::parse("28VTg95wG2zvE").expect("valid test subject id"),
            version: 1,
            kind,
            title: "Test".to_string(),
            snippet: "Test".to_string(),
            tags: Vec::new(),
            locator,
            featured: false,
            added_at: 0,
        }
    }

    /// The whole feature request in one assertion: with the toggle off (the
    /// default), a `Site`/`App` entry stays and an `External` one is hidden.
    #[test]
    fn freenet_entries_pass_the_default_filter_external_does_not() {
        // Against the NAMED CONSTANT `App` actually uses, not a literal `false`
        // — `App`'s own `use_signal` initial value isn't otherwise reachable
        // from a test, so pinning the constant is what stands in for it. A
        // mutation flipping `SHOW_EXTERNAL_BY_DEFAULT` to `true` (shipping
        // with the feature silently defeated) fails here specifically.
        assert!(passes_external_filter(
            &entry(Kind::Site),
            SHOW_EXTERNAL_BY_DEFAULT
        ));
        assert!(passes_external_filter(
            &entry(Kind::App),
            SHOW_EXTERNAL_BY_DEFAULT
        ));
        assert!(!passes_external_filter(
            &entry(Kind::External),
            SHOW_EXTERNAL_BY_DEFAULT
        ));
    }

    /// The toggle is an OVERRIDE, not a permanent removal: with it on, nothing
    /// is filtered by kind at all.
    #[test]
    fn every_kind_passes_once_the_toggle_is_on() {
        for kind in [Kind::App, Kind::Site, Kind::External] {
            assert!(
                passes_external_filter(&entry(kind), true),
                "{kind:?} must pass once show_external is true"
            );
        }
    }

    /// `kind` and `locator` are independent fields — nothing in `IndexEntry::
    /// check`, `atlasctl add`, or the on-chain contract requires them to
    /// agree — and the crawler has at least one path that mislabels a real
    /// `app:`/`freenet:` locator as `Kind::External` (an `app:` locator that
    /// fails `map_locator`'s reversal falls into a scheme-blind "external"
    /// fallback). The filter must follow `locator`, the field that actually
    /// decides where Open navigates, not `kind`, a display label — or a
    /// "Freenet only" filter could hide real Freenet content, the one
    /// failure mode this feature must never have.
    #[test]
    fn the_filter_follows_locator_not_kind_when_they_disagree() {
        let mislabeled_freenet_site = entry_with(
            Kind::External,
            Locator::Freenet {
                contract_id: "EqJ5YpEEV3XLqEvKWLQHFhGAac2qXzSUoE6k2zbdnXBr".to_string(),
                path: "/".to_string(),
            },
        );
        assert!(
            passes_external_filter(&mislabeled_freenet_site, false),
            "a Freenet locator must pass the Freenet-only filter even if its \
             `kind` says External"
        );

        let mislabeled_web_page = entry_with(
            Kind::Site,
            Locator::External {
                url: "https://example.com".to_string(),
            },
        );
        assert!(
            !passes_external_filter(&mislabeled_web_page, false),
            "an External locator must be hidden by the Freenet-only filter \
             even if its `kind` says Site"
        );
    }

    /// `App` needs a live Dioxus render to test directly, so the wiring
    /// between it and the pure functions above is otherwise unverified —
    /// exactly the gap that let each of these four call-site mutations
    /// survive every other test in this module while trying them by hand:
    /// deleting the entries-chain filter call entirely (the feature vanishes,
    /// external links reappear, nothing red); the `total_external` count
    /// reading `show_external()` instead of the hardcoded `false` (turning
    /// the toggle into a one-way trap — see the comment on that binding); the
    /// render guard's `> 0` loosened to `>= 0` (the row renders even with
    /// nothing to show, reading "0 web links hidden"); and the toggle's
    /// initial value bypassing `SHOW_EXTERNAL_BY_DEFAULT` for a bare literal
    /// (silently shipping with the whole feature defeated). Source-scraped,
    /// scoped to `App`'s own body so it cannot pass by matching code in
    /// `EntryCard` or elsewhere — the region ends at the next top-level `fn`,
    /// which is `EntryCard`'s.
    #[test]
    fn app_wires_the_filter_up_correctly() {
        let src = include_str!("main.rs");
        let production = src
            .split("\nmod tests")
            .next()
            .expect("source must have a pre-test region");
        assert!(
            !production.contains("fn app_wires_the_filter_up_correctly"),
            "the scan region must exclude the test module, or the pin matches itself"
        );
        let at = production
            .find("fn App() -> Element {")
            .expect("App must exist");
        let body = &production[at..];
        let end = body
            .find("\nfn ")
            .map(|e| at + e)
            .unwrap_or(production.len());
        let body = &production[at..end];
        assert!(
            body.contains("use_signal(|| SHOW_EXTERNAL_BY_DEFAULT)"),
            "the toggle's initial value must be the named default constant"
        );
        assert!(
            body.contains(".filter(|e| passes_external_filter(e, show_external()))"),
            "the entries grid must actually be filtered by the toggle"
        );
        // Matched as one line, not the multi-line block rustfmt actually
        // produces around it — the exact indentation/wrapping is not stable
        // across a `cargo fmt` run, but this inner expression's own text is.
        //
        // COUNT, not `contains`: this exact snippet legitimately appears
        // TWICE in a healthy `App` — once for `total_external` (the row-
        // visibility gate) and once for `external_matching_query` (the label
        // text). A bare `contains` stays satisfied by whichever one was NOT
        // mutated, so mutating either `false` to `show_external()` — turning
        // either the gate or the label into a query/toggle-dependent value —
        // would pass a `contains` check and fail to fail.
        assert_eq!(
            body.matches(".filter(|e| !passes_external_filter(e, false))")
                .count(),
            2,
            "both `total_external` and `external_matching_query` must be \
             computed with a hardcoded false, not the live toggle value — \
             using show_external() in either turns a query or the toggle \
             itself into something that can make the row (or its count) \
             behave inconsistently, including the one-way-trap case where \
             turning the toggle ON removes the only way to turn it back OFF"
        );
        assert!(
            body.contains("if total_external > 0 {"),
            "the toggle row must be gated on the query-independent, toggle- \
             independent count, or it can vanish out from under a search or \
             the toggle itself"
        );
    }

    #[test]
    fn plural_s_only_omits_on_exactly_one() {
        assert_eq!(plural_s(0), "s");
        assert_eq!(plural_s(1), "");
        assert_eq!(plural_s(2), "s");
    }
}
