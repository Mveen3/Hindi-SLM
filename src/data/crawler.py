"""
Parallel Web Corpus Crawler
===========================

Collects the manually-gathered portion of a language's corpus (the >=20% the
brief requires we crawl ourselves) from a seed list of domains.

Per-domain strategy, auto-detected in order:
  1. WordPress REST API  ``/wp-json/wp/v2/posts|pages``  — clean JSON, preferred
  2. HTML BFS from the homepage                          — fallback

Seed domains are not hard-coded here: each language supplies its own list in
``<language>/configs/data_sources.yaml``, so this engine is language-neutral.

Resumable: every site checkpoints its progress (WP page cursor, or the full
BFS frontier) beside its output, and URLs already present in the ``.jsonl``
are never written twice. Re-running the same command continues where it
stopped; ``--fresh`` starts over.

Output : ``<language>/data/raw/manual/<domain>.jsonl``
State  : ``<language>/data/raw/manual/<domain>.state.json``

Invoked through ``python main.py crawl --lang {hindi,nepali}``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
import time
from collections import deque
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse

import aiohttp
from bs4 import BeautifulSoup

# ════════════════════════════════════════════════════════════════
# Configuration  (override via CLI flags)
# ════════════════════════════════════════════════════════════════

TARGET_TOKENS   = 200_000_000
BYTES_PER_TOKEN = 10
TARGET_BYTES    = TARGET_TOKENS * BYTES_PER_TOKEN   # 1 000 000 000 bytes = 1 GB

# Set from the selected language at startup; see run().
OUTPUT_DIR      = None
CANDIDATES      = []
LOG_FILE        = "crawler.log"

SITE_WORKERS    = 12          # parallel site coroutines
PER_HOST_CONN   = 6           # max simultaneous TCP connections per domain
RATE_DELAY      = 0.20        # minimum gap (s) between requests to same host
TIMEOUT         = aiohttp.ClientTimeout(total=30, connect=8)
WP_TIMEOUT      = aiohttp.ClientTimeout(total=60, connect=10)  # WP API returns large payloads
MAX_RETRIES     = 3
WP_MAX_RETRIES  = 5           # WP endpoints need more retries (slow servers)

WP_PER_PAGE       = 100       # posts per WP API call (max allowed by WP)
MAX_HTML_PAGES    = 5_000     # BFS hard cap per non-WP site
MIN_WP_RECORDS    = 5         # WP total records below this → fall back to HTML
WP_STATE_SAVE_EVERY   = 5     # checkpoint every N WP pages
HTML_STATE_SAVE_EVERY = 50    # checkpoint every N HTML pages (was 25)
HTML_CONCURRENT   = 4         # concurrent fetch workers inside HTML BFS per site

# URL filter sets for HTML BFS
SKIP_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".ico",
    ".pdf", ".zip", ".tar", ".gz", ".mp4", ".mp3", ".ogg",
    ".css", ".js", ".xml", ".rss", ".atom",
}
SKIP_PATH_PARTS = {
    "/tag/", "/tags/", "/author/", "/authors/", "/search/",
    "/login", "/logout", "/register", "/signup", "/cart",
    "/wp-admin", "/wp-login", "/feed/", "/sitemap", "/cdn-cgi/",
    "/page/",   # category pagination — we hit API for posts instead
}

UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
HDRS = {
    "User-Agent": UA,
    "Accept-Language": "hi,en-IN;q=0.9,en;q=0.5",
    "Accept": "text/html,application/json,*/*;q=0.8",
}

# ════════════════════════════════════════════════════════════════
# Shared state
# ════════════════════════════════════════════════════════════════

class ByteCounter:
    """
    Async-safe global byte accumulator.
    Sets .done (asyncio.Event) the moment the target is reached.
    """

    def __init__(self, target: int) -> None:
        self._n     = 0
        self.target = target
        self.done   = asyncio.Event()

    async def add(self, n: int) -> None:
        # asyncio is single-threaded — no lock needed for += 
        self._n += n
        if self._n >= self.target and not self.done.is_set():
            self.done.set()

    @property
    def count(self) -> int:
        return self._n

    @property
    def tokens(self) -> float:
        return self._n / BYTES_PER_TOKEN

    @property
    def pct(self) -> float:
        return self._n / self.target * 100


class HostRateLimiter:
    """
    Enforces RATE_DELAY seconds between successive requests to the same host.
    Each host gets its own asyncio.Lock so different hosts never block each other.
    """

    def __init__(self, delay: float = RATE_DELAY) -> None:
        self._delay = delay
        self._last:  dict[str, float] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def _get_lock(self, host: str) -> asyncio.Lock:
        if host not in self._locks:
            self._locks[host] = asyncio.Lock()
        return self._locks[host]

    async def wait(self, url: str) -> None:
        host = urlparse(url).netloc
        async with self._get_lock(host):
            gap = self._delay - (time.monotonic() - self._last.get(host, 0.0))
            if gap > 0:
                await asyncio.sleep(gap)
            self._last[host] = time.monotonic()


# ════════════════════════════════════════════════════════════════
# Per-site checkpoint state  (NEW)
# ════════════════════════════════════════════════════════════════

def load_state(state_path: Path) -> dict:
    """Load a site's checkpoint. Returns {} if missing or corrupted."""
    if state_path.exists():
        try:
            return json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_state(state_path: Path, state: dict) -> None:
    """Atomic write so a Ctrl+C mid-write can never corrupt the checkpoint."""
    tmp = state_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    tmp.replace(state_path)


def load_seen_urls(out_path: Path) -> set[str]:
    """
    Read back a site's already-saved records and return the set of URLs
    already on disk, so we never fetch-and-rewrite the same article twice.
    """
    seen: set[str] = set()
    if not out_path.exists():
        return seen
    with open(out_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            u = rec.get("url")
            if u:
                seen.add(u)
    return seen


# ════════════════════════════════════════════════════════════════
# HTML helpers
# ════════════════════════════════════════════════════════════════

_RE_TAG   = re.compile(r"<[^>]+>")
_RE_SPACE = re.compile(r"\s+")

def strip_tags(fragment: str) -> str:
    """Convert an HTML fragment to plain text (single line).
    Uses fast regex for simple HTML; falls back to BS4 only for entities."""
    if not fragment:
        return ""
    text = _RE_TAG.sub(" ", fragment)
    text = _RE_SPACE.sub(" ", text).strip()
    # Decode common HTML entities
    if "&" in text:
        import html as _html
        text = _html.unescape(text)
    return text


_CONTENT_SELECTORS = [
    "article",
    "[class*='post-content']",
    "[class*='entry-content']",
    "[class*='article-body']",
    "[class*='story-body']",
    "[class*='td-post-content']",
    "[class*='content-area']",
    "[id='content']",
    "main",
]
_JUNK_TAGS = ["script", "style", "nav", "footer", "header",
              "aside", "form", "noscript", "figure", "figcaption", "iframe"]


def extract_main_text(html: str) -> str:
    """Pull the main readable body from a full HTML page."""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(_JUNK_TAGS):
        tag.decompose()

    el: Optional[object] = None
    for sel in _CONTENT_SELECTORS:
        candidate = soup.select_one(sel)
        if candidate and len(candidate.get_text(strip=True)) > 300:
            el = candidate
            break

    node = el or soup.find("body") or soup
    text = node.get_text("\n")  # type: ignore[union-attr]
    lines = [ln.strip() for ln in text.splitlines() if len(ln.strip()) > 25]
    return "\n".join(lines)


def same_domain_links(html: str, page_url: str, base: str) -> list[str]:
    """Return de-duplicated internal URLs that look like article pages."""
    base_host = urlparse(base).netloc
    seen: set[str] = set()
    out:  list[str] = []

    for a in BeautifulSoup(html, "lxml").find_all("a", href=True):
        href = a["href"].strip()
        full = urljoin(page_url, href).split("#")[0].split("?")[0]
        p    = urlparse(full)

        if p.netloc != base_host:
            continue
        if p.scheme not in ("http", "https"):
            continue
        path = p.path.lower()
        if any(path.endswith(e) for e in SKIP_EXTENSIONS):
            continue
        if any(s in path for s in SKIP_PATH_PARTS):
            continue

        norm = full.rstrip("/")
        if norm and norm not in seen:
            seen.add(norm)
            out.append(norm)

    return out


# ════════════════════════════════════════════════════════════════
# Network helper
# ════════════════════════════════════════════════════════════════

async def http_get(
    session: aiohttp.ClientSession,
    url: str,
    rl: HostRateLimiter,
    mode: str = "text",         # "text" | "json" | "raw" (returns ClientResponse)
    retries: int = MAX_RETRIES,
) -> tuple[int, object]:
    """
    Async GET with exponential-backoff retries.
    Returns (http_status, body) where body is str | list | dict | None.
    """
    for attempt in range(retries):
        try:
            await rl.wait(url)
            async with session.get(
                url, headers=HDRS, timeout=TIMEOUT,
                ssl=False, allow_redirects=True,
            ) as r:
                if r.status == 200:
                    if mode == "json":
                        return 200, await r.json(content_type=None)
                    return 200, await r.text(errors="replace")
                if r.status in (400, 403, 404, 410, 451):
                    return r.status, None          # permanent → skip silently
                if r.status == 429:
                    logging.debug(f"429 on {url} — cooling 15 s")
                    await asyncio.sleep(15)
                    continue
                if r.status >= 500:
                    await asyncio.sleep(3 * (attempt + 1))
                    continue
                return r.status, None
        except asyncio.TimeoutError:
            await asyncio.sleep(2 ** attempt)
        except aiohttp.ClientError as exc:
            logging.debug(f"ClientError {url}: {exc}")
            await asyncio.sleep(2 ** attempt)
        except Exception as exc:
            logging.debug(f"Unexpected {url}: {exc}")
            await asyncio.sleep(2 ** attempt)

    return 0, None   # exhausted all retries


# ════════════════════════════════════════════════════════════════
# Mode 1 — WordPress REST API
# ════════════════════════════════════════════════════════════════

async def wp_detect(
    session: aiohttp.ClientSession, base: str, rl: HostRateLimiter
) -> bool:
    """Return True if the site serves the WordPress REST API."""
    probe = base.rstrip("/") + "/wp-json/wp/v2/posts?per_page=1"
    st, data = await http_get(session, probe, rl, "json")
    return st == 200 and isinstance(data, list)


async def wp_crawl(
    session: aiohttp.ClientSession,
    base: str,
    counter: ByteCounter,
    out,                         # open writable file
    rl: HostRateLimiter,
    log: logging.Logger,
    seen_urls: set[str],
    state: dict,
    state_path: Path,
) -> int:
    """
    Paginate through /wp-json/wp/v2/posts and /wp-json/wp/v2/pages.
    Uses X-WP-TotalPages header to stop exactly when done.
    Resumable: reads/writes state["wp"][resource]["next_page"] so a
    restart continues from the exact page it left off on, per resource.
    Returns the number of NEW records written this run.
    """
    api = base.rstrip("/") + "/wp-json/wp/v2"
    wp_state = state.setdefault("wp", {})
    new_records = 0

    for resource in ("posts", "pages"):
        if counter.done.is_set():
            break

        res_state = wp_state.setdefault(
            resource, {"next_page": 1, "total_pages": None, "done": False}
        )
        if res_state.get("done"):
            continue   # this resource was already fully paginated in a past run

        page        = res_state["next_page"]
        total_pages = res_state["total_pages"]

        while not counter.done.is_set():
            url = (
                f"{api}/{resource}"
                f"?per_page={WP_PER_PAGE}&page={page}&_embed=0"
            )

            data: Optional[list] = None
            fetch_ok = False

            for attempt in range(WP_MAX_RETRIES):
                try:
                    await rl.wait(url)
                    async with session.get(
                        url, headers=HDRS, timeout=WP_TIMEOUT, ssl=False
                    ) as r:
                        if r.status == 400:          # WP returns 400 beyond last page
                            data, fetch_ok = [], True
                            res_state["done"] = True
                            break
                        if r.status != 200:
                            log.debug(
                                f"wp/{resource} page={page} "
                                f"attempt={attempt+1} → HTTP {r.status}"
                            )
                            await asyncio.sleep(3 * (attempt + 1))
                            continue
                        if total_pages is None:
                            raw_tp = r.headers.get("X-WP-TotalPages", "1")
                            total_pages = max(1, int(raw_tp))
                            res_state["total_pages"] = total_pages
                        data = await r.json(content_type=None)
                        fetch_ok = True
                        break
                except Exception as exc:
                    # was: log.warning(f"... fetch error: {exc}") which prints
                    # blank text for some aiohttp errors — always show the type.
                    log.warning(
                        f"wp/{resource} p{page} fetch error "
                        f"(attempt {attempt+1}/{WP_MAX_RETRIES}): "
                        f"{type(exc).__name__}: {exc!r}"
                    )
                    await asyncio.sleep(2 ** attempt)

            if not fetch_ok:
                log.warning(
                    f"wp/{resource} skipping page {page} for {base} "
                    f"after {WP_MAX_RETRIES} failed attempts — advancing "
                    f"to page {page+1} for next attempt"
                )
                # CRITICAL FIX: advance past the failed page so we don't
                # get stuck retrying the same page forever on every run.
                page += 1
                res_state["next_page"] = page
                save_state(state_path, state)
                continue   # try the next page immediately

            if res_state.get("done"):
                break   # hit the "past last page" 400

            if not data:
                res_state["done"] = True
                break

            for post in data:
                if counter.done.is_set():
                    break
                try:
                    link = post.get("link", "")
                    if link and link in seen_urls:
                        continue   # already saved in a previous run

                    title   = strip_tags(post.get("title",   {}).get("rendered", ""))
                    content = strip_tags(post.get("content", {}).get("rendered", ""))
                    excerpt = strip_tags(post.get("excerpt", {}).get("rendered", ""))
                    date    = post.get("date", "")

                    body = f"{title}\n\n{content}".strip()
                    if len(body) < 80:           # skip stubs
                        continue

                    rec  = {
                        "url": link, "source": base, "date": date,
                        "title": title, "excerpt": excerpt, "text": body,
                    }
                    line = json.dumps(rec, ensure_ascii=False) + "\n"
                    out.write(line)
                    await counter.add(len(line.encode("utf-8")))
                    if link:
                        seen_urls.add(link)
                    new_records += 1

                except Exception as exc:
                    log.debug(f"post parse error: {exc}")

            log.info(
                f"[WP/{resource:5}] {base}  "
                f"page {page:>4}/{total_pages or '?':>4}  "
                f"{counter.tokens/1e6:.2f}M tok  {counter.pct:.1f}%"
            )

            page += 1
            res_state["next_page"] = page
            if page % WP_STATE_SAVE_EVERY == 0:
                save_state(state_path, state)

            if total_pages and page > total_pages:
                res_state["done"] = True
                break

        save_state(state_path, state)

    return new_records


# ════════════════════════════════════════════════════════════════
# Mode 2 — BFS HTML scraper (fallback)
# ════════════════════════════════════════════════════════════════

async def html_crawl(
    session: aiohttp.ClientSession,
    base: str,
    counter: ByteCounter,
    out,                         # open writable file
    rl: HostRateLimiter,
    log: logging.Logger,
    seen_urls: set[str],
    state: dict,
    state_path: Path,
) -> int:
    """
    Breadth-first crawl of internal links with concurrent fetch workers.
    Extracts article text from each page and saves as JSONL.
    Uses HTML_CONCURRENT parallel workers to fetch pages from the queue.
    Returns the number of NEW records written this run.
    """
    html_state = state.setdefault("html", {})
    if html_state.get("done"):
        return 0

    visited: set[str]   = set(html_state.get("visited", []))
    queue:   deque[str] = deque(html_state.get("queue", []))
    crawled = html_state.get("crawled", 0)
    new_records = 0
    # Lock to protect shared mutable state (queue, visited, crawled, out)
    lock = asyncio.Lock()

    start = base.rstrip("/")
    if not queue and not visited:
        queue.append(start)
        visited.add(start)
    visited |= seen_urls

    _title_re = re.compile(r"<title[^>]*>(.*?)</title>", re.I | re.S)

    # --- Sitemap discovery: seed the queue with URLs from sitemap ---
    try:
        sitemap_url = base.rstrip("/") + "/sitemap.xml"
        sm_st, sm_body = await http_get(session, sitemap_url, rl, "text", retries=1)
        if sm_st == 200 and sm_body and "<loc>" in sm_body:
            import xml.etree.ElementTree as ET
            root = ET.fromstring(sm_body)
            ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
            locs = [el.text.strip() for el in root.iter()
                    if el.tag.endswith("}loc") or el.tag == "loc"]
            added = 0
            base_host = urlparse(base).netloc
            for loc in locs:
                norm = loc.rstrip("/")
                if urlparse(norm).netloc == base_host and norm not in visited:
                    # Check if it's a sub-sitemap
                    if norm.endswith(".xml"):
                        # Fetch sub-sitemap
                        ss_st, ss_body = await http_get(session, norm, rl, "text", retries=1)
                        if ss_st == 200 and ss_body and "<loc>" in ss_body:
                            try:
                                sub_root = ET.fromstring(ss_body)
                                for sub_el in sub_root.iter():
                                    if sub_el.tag.endswith("}loc") or sub_el.tag == "loc":
                                        sub_loc = sub_el.text.strip().rstrip("/")
                                        if (urlparse(sub_loc).netloc == base_host
                                                and sub_loc not in visited
                                                and not sub_loc.endswith(".xml")):
                                            visited.add(sub_loc)
                                            queue.append(sub_loc)
                                            added += 1
                            except ET.ParseError:
                                pass
                    else:
                        visited.add(norm)
                        queue.append(norm)
                        added += 1
            if added:
                log.info(f"[SMAP ] {base}  seeded {added} URLs from sitemap")
    except Exception:
        pass   # sitemap is optional, BFS will work without it

    homepage_failed = False

    async def _worker():
        nonlocal crawled, new_records, homepage_failed
        while True:
            if counter.done.is_set():
                return

            async with lock:
                if not queue or crawled >= MAX_HTML_PAGES:
                    return
                url = queue.popleft()
                my_crawl_num = crawled

            st, html = await http_get(session, url, rl, "text")
            if st != 200 or not html:
                async with lock:
                    if crawled == 0 and not homepage_failed:
                        homepage_failed = True
                        log.warning(
                            f"[HTML ] {base}  homepage fetch failed "
                            f"(status={st}) — site may be blocking crawlers, "
                            f"down, or redirecting somewhere unexpected"
                        )
                continue

            # Extract text and discover links
            body = None
            links = []
            if url not in seen_urls:
                body = extract_main_text(html)
            links = same_domain_links(html, url, base)

            async with lock:
                crawled += 1

                if body and len(body) > 200 and url not in seen_urls:
                    m     = _title_re.search(html)
                    title = strip_tags(m.group(1)) if m else ""
                    rec   = {"url": url, "source": base, "title": title, "text": body}
                    line  = json.dumps(rec, ensure_ascii=False) + "\n"
                    out.write(line)
                    await counter.add(len(line.encode("utf-8")))
                    seen_urls.add(url)
                    new_records += 1

                for link in links:
                    if link not in visited:
                        visited.add(link)
                        queue.append(link)

                if crawled % 100 == 0:
                    log.info(
                        f"[HTML ] {base}  "
                        f"pages {crawled:>4}/{MAX_HTML_PAGES}  "
                        f"{counter.tokens/1e6:.2f}M tok  {counter.pct:.1f}%"
                    )

                if crawled % HTML_STATE_SAVE_EVERY == 0:
                    html_state.update({
                        "visited": list(visited), "queue": list(queue),
                        "crawled": crawled,
                    })
                    save_state(state_path, state)

    # Run workers concurrently
    workers = [asyncio.create_task(_worker()) for _ in range(HTML_CONCURRENT)]
    await asyncio.gather(*workers, return_exceptions=True)

    if crawled == 0:
        log.warning(
            f"[HTML ] {base}  finished with 0 pages crawled — "
            f"homepage never returned usable content, see warning above"
        )
    elif new_records == 0 and crawled < 10:
        log.warning(
            f"[HTML ] {base}  crawled {crawled} page(s) but extracted "
            f"0 new records — page bodies may be too short, behind a "
            f"paywall/JS wall, or the site has few internal links to follow"
        )

    if not queue or crawled >= MAX_HTML_PAGES:
        html_state["done"] = True

    html_state.update({
        "visited": list(visited), "queue": list(queue), "crawled": crawled,
    })
    save_state(state_path, state)

    return new_records


# ════════════════════════════════════════════════════════════════
# Per-site orchestrator
# ════════════════════════════════════════════════════════════════

async def crawl_site(
    session: aiohttp.ClientSession,
    url: str,
    counter: ByteCounter,
    rl: HostRateLimiter,
    sem: asyncio.Semaphore,
    args: argparse.Namespace,
) -> None:
    if counter.done.is_set():
        return

    async with sem:
        log  = logging.getLogger(urlparse(url).netloc)
        slug = urlparse(url).netloc.replace(".", "_")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path   = OUTPUT_DIR / f"{slug}.jsonl"
        state_path = OUTPUT_DIR / f"{slug}.state.json"

        if args.fresh:
            state: dict = {}
            seen_urls: set[str] = set()
            if state_path.exists():
                state_path.unlink()
            open_mode = "w"
        else:
            state = load_state(state_path)
            if state.get("site_done"):
                log.info(f"⏭ already complete — skipping  {url}")
                return
            seen_urls = load_seen_urls(out_path)
            if seen_urls:
                log.info(f"↺ resuming — {len(seen_urls)} record(s) already saved  {url}")
            open_mode = "a"

        try:
            # line-buffered (buffering=1) so every \n flushes to disk
            with open(out_path, open_mode, encoding="utf-8", buffering=1) as f:
                mode = state.get("mode")

                if mode is None:
                    is_wp = await wp_detect(session, url, rl)
                    mode = "wp" if is_wp else "html"
                    state["mode"] = mode
                    save_state(state_path, state)

                total_new = 0

                if mode == "wp":
                    log.info(f"✔ WordPress API  {url}")
                    total_new += await wp_crawl(
                        session, url, counter, f, rl, log, seen_urls, state, state_path
                    )

                    wp_state  = state.get("wp", {})
                    both_done = all(
                        wp_state.get(r, {}).get("done") for r in ("posts", "pages")
                    )

                    if len(seen_urls) < args.min_wp_records and not counter.done.is_set():
                        log.warning(
                            f"⚠ WP API yielded only {len(seen_urls)} record(s) "
                            f"for {url} — falling back to HTML BFS"
                        )
                        state["mode"] = "html"
                        save_state(state_path, state)
                        log.info(f"↻ HTML BFS       {url}")
                        total_new += await html_crawl(
                            session, url, counter, f, rl, log, seen_urls, state, state_path
                        )
                        if state.get("html", {}).get("done"):
                            state["site_done"] = True
                            save_state(state_path, state)
                    elif both_done:
                        state["site_done"] = True
                        save_state(state_path, state)

                else:
                    log.info(f"↻ HTML BFS       {url}")
                    total_new += await html_crawl(
                        session, url, counter, f, rl, log, seen_urls, state, state_path
                    )
                    if state.get("html", {}).get("done"):
                        state["site_done"] = True
                        save_state(state_path, state)

            log.info(
                f"✓ finished  {url}  "
                f"(+{total_new} new this run, {len(seen_urls)} total saved)"
            )

        except asyncio.CancelledError:
            save_state(state_path, state)
            log.info(f"↩ cancelled — progress saved  {url}")

        except Exception as exc:
            save_state(state_path, state)
            log.exception(f"✗ fatal  {url}  →  {exc}")


# ════════════════════════════════════════════════════════════════
# Progress ticker
# ════════════════════════════════════════════════════════════════

async def progress_ticker(counter: ByteCounter, start_bytes: int) -> None:
    start = time.monotonic()
    while not counter.done.is_set():
        await asyncio.sleep(60)
        t         = time.monotonic() - start
        new_bytes = counter.count - start_bytes
        kbps      = new_bytes / t / 1024 if t > 0 else 0
        rem       = TARGET_BYTES - counter.count
        eta       = rem / (kbps * 1024) / 60 if kbps > 0 else float("inf")
        print(
            f"\n{'─'*58}\n"
            f"  Tokens  : {counter.tokens/1e6:>9.3f} M  /  {TARGET_TOKENS/1e6:.0f} M"
            f"   [{counter.pct:5.2f}%]\n"
            f"  Data    : {counter.count/1e9:.4f} GB\n"
            f"  Rate    : {kbps:,.1f} KB/s  (this session)\n"
            f"  ETA     : {eta:.0f} min\n"
            f"{'─'*58}",
            flush=True,
        )


# ════════════════════════════════════════════════════════════════
# End-of-run summary  (NEW)
# ════════════════════════════════════════════════════════════════

def print_site_summary(log: logging.Logger) -> None:
    rows = []
    for state_path in sorted(OUTPUT_DIR.glob("*.state.json")):
        try:
            st = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        slug   = state_path.name[: -len(".state.json")]
        jsonl  = OUTPUT_DIR / f"{slug}.jsonl"
        nbytes = jsonl.stat().st_size if jsonl.exists() else 0
        status = "done" if st.get("site_done") else "partial"
        rows.append((slug, status, st.get("mode", "?"), nbytes))

    if not rows:
        return

    log.info("── per-site status (re-run the script to continue any 'partial') ──")
    for slug, status, mode, nbytes in rows:
        log.info(f"  {slug:35} {status:8} mode={mode or '?':5} {nbytes/1e6:8.2f} MB")


# ════════════════════════════════════════════════════════════════
# Entry point
# ════════════════════════════════════════════════════════════════

async def _crawl(args) -> None:
    """Run the crawl for the already-configured language."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)-32s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler(OUTPUT_DIR / LOG_FILE, "a", "utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    log = logging.getLogger("main")

    existing_bytes = 0
    if not args.fresh:
        existing_bytes = sum(p.stat().st_size for p in OUTPUT_DIR.glob("*.jsonl"))

    log.info(
        f"Language     : {args.language}\n"
        f"Target       : {TARGET_TOKENS:,} tokens  =  {TARGET_BYTES/1e9:.1f} GB\n"
        f"Sites        : {len(CANDIDATES)}\n"
        f"Workers      : {args.workers}\n"
        f"Rate         : {args.rate} s / host\n"
        f"Min WP recs  : {args.min_wp_records}  (below this → HTML fallback)\n"
        f"Output       : {OUTPUT_DIR.resolve()}\n"
        f"Mode         : {'FRESH (ignoring old data/state)' if args.fresh else 'RESUME'}\n"
        + (f"Already have : {existing_bytes/1e9:.4f} GB "
           f"({existing_bytes/BYTES_PER_TOKEN/1e6:.2f} M tokens)\n"
           if existing_bytes else "")
    )

    counter = ByteCounter(TARGET_BYTES)
    if existing_bytes:
        await counter.add(existing_bytes)
        if counter.done.is_set():
            log.info("Target already met by previously-saved data — nothing to do.")

    rl  = HostRateLimiter(args.rate)
    sem = asyncio.Semaphore(args.workers)

    conn = aiohttp.TCPConnector(
        limit=args.workers * PER_HOST_CONN,
        limit_per_host=PER_HOST_CONN,
        ssl=False,
        ttl_dns_cache=300,
        enable_cleanup_closed=True,
        force_close=False,
    )

    async with aiohttp.ClientSession(connector=conn) as session:
        ticker = asyncio.create_task(progress_ticker(counter, existing_bytes))
        tasks  = [
            asyncio.create_task(crawl_site(session, url, counter, rl, sem, args))
            for url in CANDIDATES
        ]

        # Bundle every site task into ONE future. Without this, asyncio.wait()
        # below would return the instant *any single* site finished (e.g. a
        # small site running out of pages in a few seconds) and cancel every
        # other in-progress site along with it — which is why the crawl was
        # dying at ~0.2M tokens instead of running until the target was hit.
        all_sites_done = asyncio.gather(*tasks, return_exceptions=True)
        stop = asyncio.create_task(counter.done.wait())

        # Stop as soon as target is hit OR all sites are exhausted
        await asyncio.wait([stop, all_sites_done], return_when=asyncio.FIRST_COMPLETED)

        # Graceful shutdown — every in-flight site saves its checkpoint
        # via the asyncio.CancelledError handler in crawl_site.
        for t in tasks:
            t.cancel()
        ticker.cancel()
        await asyncio.gather(*tasks, ticker, return_exceptions=True)

    print_site_summary(log)

    log.info(
        f"\n{'═'*58}\n"
        f"  DONE\n"
        f"  Tokens collected : {counter.tokens/1e6:.3f} M"
        f"  ({counter.count:,} bytes)\n"
        f"  Output directory : {OUTPUT_DIR.resolve()}\n"
        f"  Re-run the same command any time to continue any 'partial' sites.\n"
        f"{'═'*58}"
    )


def run(lang, workers: int = SITE_WORKERS, rate: float = RATE_DELAY,
        output: str = None, min_wp_records: int = MIN_WP_RECORDS,
        fresh: bool = False) -> None:
    """
    Crawl one language's seed domains into its raw/manual directory.

    Args:
        lang:           Language to crawl; supplies both the seed list and the
                        output directory.
        workers:        Parallel site workers.
        rate:           Minimum seconds between requests to the same host.
        output:         Override for ``<language>/data/raw/manual``.
        min_wp_records: Below this many WordPress records, fall back to HTML BFS.
        fresh:          Ignore saved state and re-crawl every site.
    """
    global OUTPUT_DIR, CANDIDATES
    CANDIDATES = list(lang.load_sources()["crawl_seeds"])
    OUTPUT_DIR = Path(output) if output else lang.raw_dir / "manual"

    args = argparse.Namespace(
        language=f"{lang.display} ({lang.model_label})",
        workers=workers, rate=rate, min_wp_records=min_wp_records, fresh=fresh,
    )

    try:
        asyncio.run(_crawl(args))
    except KeyboardInterrupt:
        print("\n[Ctrl+C] Interrupted — progress saved, safe to resume by re-running.")


def main():
    parser = argparse.ArgumentParser(description="Crawl seed domains for raw corpus collection.")
    from src import languages
    languages.add_language_arg(parser)
    parser.add_argument("--workers", type=int, default=SITE_WORKERS,
                        help="Parallel site workers (default: %(default)s).")
    parser.add_argument("--rate", type=float, default=RATE_DELAY,
                        help="Min seconds between requests to one host (default: %(default)s).")
    parser.add_argument("--output", default=None,
                        help="Override <language>/data/raw/manual.")
    parser.add_argument("--min-wp-records", type=int, default=MIN_WP_RECORDS,
                        help="Below this many WordPress records, fall back to HTML BFS.")
    parser.add_argument("--fresh", action="store_true",
                        help="Ignore saved state and re-crawl every site.")
    args = parser.parse_args()

    run(
        languages.get(args.lang),
        workers=args.workers,
        rate=args.rate,
        output=args.output,
        min_wp_records=args.min_wp_records,
        fresh=args.fresh,
    )


if __name__ == "__main__":
    main()

