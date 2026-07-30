#![allow(non_snake_case)]
//! Atlas Discover: a read-only front door to Freenet. Connects to the local node
//! over the WebSocket command API, GET+SUBSCRIBEs the index contract, and renders
//! browse + client-side search + Open. No identity, no writes, no delegate.

use std::cell::RefCell;

use atlas_common::{IndexEntry, IndexState, Kind, Locator};
use dioxus::prelude::*;
use freenet_stdlib::client_api::{
    ClientRequest, ContractRequest, ContractResponse, Error, HostResponse, WebApi,
};
use freenet_stdlib::prelude::ContractInstanceId;
use wasm_bindgen::JsValue;
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

static STATE: GlobalSignal<Option<IndexState>> = Signal::global(|| None);
static STATUS: GlobalSignal<String> = Signal::global(|| "connecting…".to_string());

thread_local! {
    static API: RefCell<Option<WebApi>> = const { RefCell::new(None) };
}

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
.empty { color:var(--dim); padding:3rem 0; text-align:center; }
.foot { margin-top:2.5rem; padding-top:1.2rem; border-top:1px solid var(--line);
  color:var(--faint); font-size:.76rem; line-height:1.6; }
"#;

fn main() {
    #[cfg(target_arch = "wasm32")]
    std::panic::set_hook(Box::new(|info| {
        web_sys::console::error_1(&format!("atlas panic: {info}").into());
    }));
    launch(App);
}

fn App() -> Element {
    use_hook(|| {
        set_shell_title("Atlas");
        connect();
    });
    let mut query = use_signal(String::new);
    let q = query().to_lowercase();

    // Total live entries (unfiltered) for the header count, and the filtered set
    // shown in the grid.
    let total = STATE
        .read()
        .as_ref()
        .map(|s| s.live_entries().count())
        .unwrap_or(0);
    let entries: Vec<IndexEntry> = match STATE.read().as_ref() {
        Some(state) => {
            let mut v: Vec<IndexEntry> = state.live_entries().cloned().collect();
            v.sort_by(|a, b| {
                b.featured
                    .cmp(&a.featured)
                    .then(b.added_at.cmp(&a.added_at))
            });
            v.into_iter().filter(|e| matches_query(e, &q)).collect()
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
                    if STATE.read().is_some() { "Nothing matches." } else { "Loading…" }
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

fn kind_label(kind: Kind) -> &'static str {
    match kind {
        Kind::App => "app",
        Kind::Site => "site",
        Kind::External => "web",
    }
}

fn matches_query(e: &IndexEntry, q: &str) -> bool {
    if q.is_empty() {
        return true;
    }
    e.title.to_lowercase().contains(q)
        || e.snippet.to_lowercase().contains(q)
        || e.tags.iter().any(|t| t.to_lowercase().contains(q))
}
