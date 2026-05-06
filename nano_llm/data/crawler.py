"""
Multi-source web crawler with parallel fetching.

Sources:
  - Wikipedia      : random articles via Wikipedia API (free, no key)
  - Gutenberg      : public domain books via gutenberg.org (free)
  - Common Crawl   : WARC index samples via commoncrawl.org (free, S3)
  - Archive.org    : texts via Internet Archive API (free, no key)
  - RSS feeds      : any public RSS/Atom feed

Each source runs in its own daemon thread with its own rate limiter.
A shared thread-safe queue buffers fetched (title, text) pairs so the
continuous trainer can drain it without waiting for crawlers.
"""
import json
import logging
import queue
import random
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional
from urllib.parse import urljoin, urlparse

logger = logging.getLogger(__name__)

# ── Rate limiter ──────────────────────────────────────────────────────

class RateLimiter:
    """Per-source token-bucket rate limiter."""

    def __init__(self, min_interval: float = 1.0):
        self._min = min_interval
        self._last = 0.0
        self._lock = threading.Lock()

    def wait(self):
        with self._lock:
            elapsed = time.time() - self._last
            if elapsed < self._min:
                time.sleep(self._min - elapsed)
            self._last = time.time()


# ── HTTP helper ───────────────────────────────────────────────────────

def _make_session(user_agent: str = "NanoLLM-Crawler/1.0 (educational)"):
    import requests
    s = requests.Session()
    s.headers.update({"User-Agent": user_agent})
    return s


def _fetch(session, url: str, retries: int = 3, timeout: int = 30) -> Optional[str]:
    for attempt in range(retries):
        try:
            r = session.get(url, timeout=timeout)
            r.raise_for_status()
            return r.text
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                logger.debug(f"Failed to fetch {url}: {e}")
    return None


# ── Individual source crawlers ────────────────────────────────────────

class WikipediaCrawler:
    """
    Fetches random Wikipedia articles using the public API.
    Rate limit: 1 req/sec recommended by Wikimedia.
    """

    NAME = "wikipedia"
    DEFAULT_RATE = 1.0

    def __init__(self, rate_limit: float = DEFAULT_RATE):
        self._rl = RateLimiter(rate_limit)
        self._session = _make_session()

    def stream(self) -> Iterator[tuple[str, str]]:
        """Yield (title, text) indefinitely."""
        while True:
            try:
                self._rl.wait()
                # Get a random article ID
                api = (
                    "https://en.wikipedia.org/w/api.php"
                    "?action=query&format=json&list=random&rnlimit=1&rnnamespace=0"
                )
                raw = _fetch(self._session, api)
                if not raw:
                    continue
                data = json.loads(raw)
                page_id = data["query"]["random"][0]["id"]
                title   = data["query"]["random"][0]["title"]

                self._rl.wait()
                content_api = (
                    f"https://en.wikipedia.org/w/api.php"
                    f"?action=query&format=json&pageids={page_id}"
                    f"&prop=extracts&explaintext=1&exsectionformat=plain"
                )
                raw2 = _fetch(self._session, content_api)
                if not raw2:
                    continue
                text = (
                    json.loads(raw2)["query"]["pages"]
                    [str(page_id)].get("extract", "")
                )
                if text and len(text) > 500:
                    yield title, text

            except Exception as e:
                logger.warning(f"[Wikipedia] {e}")
                time.sleep(2)


class GutenbergCrawler:
    """
    Downloads plain-text books from Project Gutenberg.
    Cycles through a curated list of public-domain IDs.
    Rate limit: 2 sec between requests (Gutenberg ToS).
    """

    NAME = "gutenberg"
    DEFAULT_RATE = 2.0

    # ~60 well-known public-domain books
    BOOK_IDS = [
        1342, 11, 84, 1661, 2701, 98, 1952, 174, 345, 1260,
        16, 74, 76, 1400, 2600, 46, 1232, 2554, 4300, 5200,
        219, 768, 1184, 2148, 730, 1727, 3207, 2097, 514, 244,
        1080, 43, 35, 36, 37, 38, 45, 55, 62, 64,
        120, 135, 140, 145, 158, 161, 164, 215, 236, 238,
        2591, 2814, 3825, 4517, 5740, 6130, 7370, 8800, 9296, 11231,
    ]

    def __init__(self, rate_limit: float = DEFAULT_RATE):
        self._rl = RateLimiter(rate_limit)
        self._session = _make_session()
        self._ids = self.BOOK_IDS.copy()
        random.shuffle(self._ids)
        self._idx = 0

    def stream(self) -> Iterator[tuple[str, str]]:
        while True:
            book_id = self._ids[self._idx % len(self._ids)]
            self._idx += 1

            self._rl.wait()
            text = None
            for url_tmpl in [
                f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt",
                f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt",
            ]:
                text = _fetch(self._session, url_tmpl)
                if text and len(text) > 5000:
                    break

            if not text:
                continue

            title_m = re.search(r"Title:\s*(.+)", text)
            title = title_m.group(1).strip() if title_m else f"Book {book_id}"
            text  = self._strip_gutenberg(text)

            if len(text) > 5000:
                yield title, text

            # Reshuffle when we've gone through all IDs
            if self._idx % len(self._ids) == 0:
                random.shuffle(self._ids)

    @staticmethod
    def _strip_gutenberg(text: str) -> str:
        start = re.search(
            r"\*\*\* START OF (THIS|THE) PROJECT GUTENBERG", text, re.IGNORECASE
        )
        if start:
            text = text[start.end():]
        end = re.search(
            r"\*\*\* END OF (THIS|THE) PROJECT GUTENBERG", text, re.IGNORECASE
        )
        if end:
            text = text[: end.start()]
        return text.strip()


class CommonCrawlCrawler:
    """
    Samples text from Common Crawl via the CC Index API.
    Fetches WARC record snippets — no S3 download needed.
    Rate limit: 1 req/sec (polite default).
    """

    NAME = "commoncrawl"
    DEFAULT_RATE = 1.5

    # Recent crawl IDs — update periodically
    CRAWL_IDS = [
        "CC-MAIN-2024-10",
        "CC-MAIN-2023-50",
        "CC-MAIN-2023-40",
    ]

    # Seed domains known to have clean English text
    SEED_DOMAINS = [
        "en.wikipedia.org",
        "www.bbc.com",
        "www.theguardian.com",
        "arstechnica.com",
        "www.scientificamerican.com",
        "www.smithsonianmag.com",
        "www.nationalgeographic.com",
        "www.nature.com",
        "www.newscientist.com",
        "plato.stanford.edu",
    ]

    def __init__(self, rate_limit: float = DEFAULT_RATE):
        self._rl = RateLimiter(rate_limit)
        self._session = _make_session()

    def stream(self) -> Iterator[tuple[str, str]]:
        while True:
            crawl = random.choice(self.CRAWL_IDS)
            domain = random.choice(self.SEED_DOMAINS)
            try:
                self._rl.wait()
                index_url = (
                    f"https://index.commoncrawl.org/{crawl}-index"
                    f"?url={domain}/*&output=json&limit=5"
                )
                raw = _fetch(self._session, index_url)
                if not raw:
                    continue

                records = [json.loads(l) for l in raw.strip().splitlines() if l.strip()]
                if not records:
                    continue

                record = random.choice(records)
                text = self._fetch_warc_text(record)
                if text and len(text) > 300:
                    title = record.get("url", domain)
                    yield title, text

            except Exception as e:
                logger.debug(f"[CommonCrawl] {e}")
                time.sleep(3)

    def _fetch_warc_text(self, record: dict) -> Optional[str]:
        """Fetch the actual page text from the WARC record offset."""
        try:
            filename = record["filename"]
            offset   = int(record["offset"])
            length   = int(record["length"])

            self._rl.wait()
            url = f"https://data.commoncrawl.org/{filename}"
            headers = {"Range": f"bytes={offset}-{offset + length - 1}"}
            r = self._session.get(url, headers=headers, timeout=30)
            r.raise_for_status()

            # Decompress WARC gzip chunk
            import gzip, io
            try:
                content = gzip.decompress(r.content).decode("utf-8", errors="replace")
            except Exception:
                content = r.content.decode("utf-8", errors="replace")

            # Extract HTTP body (after double CRLF)
            body_start = content.find("\r\n\r\n")
            if body_start != -1:
                content = content[body_start + 4:]

            return self._html_to_text(content)
        except Exception as e:
            logger.debug(f"[CommonCrawl] WARC fetch error: {e}")
            return None

    @staticmethod
    def _html_to_text(html: str) -> str:
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
                tag.decompose()
            lines = [l.strip() for l in soup.get_text("\n").splitlines() if l.strip()]
            return "\n".join(lines)
        except ImportError:
            text = re.sub(r"<[^>]+>", " ", html)
            return re.sub(r"\s+", " ", text).strip()


class ArchiveOrgCrawler:
    """
    Fetches plain-text items from the Internet Archive.
    Uses the public search API — no key required.
    Rate limit: 1 req/sec.
    """

    NAME = "archive_org"
    DEFAULT_RATE = 1.5

    COLLECTIONS = [
        "gutenberg",
        "americana",
        "universallibrary",
        "biodiversity",
        "nasa",
    ]

    def __init__(self, rate_limit: float = DEFAULT_RATE):
        self._rl = RateLimiter(rate_limit)
        self._session = _make_session()

    def stream(self) -> Iterator[tuple[str, str]]:
        while True:
            collection = random.choice(self.COLLECTIONS)
            try:
                self._rl.wait()
                # Search for text items in the collection
                search_url = (
                    "https://archive.org/advancedsearch.php"
                    f"?q=collection:{collection}+mediatype:texts+language:English"
                    "&fl[]=identifier,title&sort[]=random&rows=5&output=json"
                )
                raw = _fetch(self._session, search_url)
                if not raw:
                    continue

                docs = json.loads(raw).get("response", {}).get("docs", [])
                if not docs:
                    continue

                item = random.choice(docs)
                identifier = item.get("identifier", "")
                title      = item.get("title", identifier)

                self._rl.wait()
                # Get item metadata to find a .txt file
                meta_url = f"https://archive.org/metadata/{identifier}"
                meta_raw = _fetch(self._session, meta_url)
                if not meta_raw:
                    continue

                meta = json.loads(meta_raw)
                files = meta.get("files", [])
                txt_files = [
                    f for f in files
                    if f.get("name", "").endswith(".txt")
                    and int(f.get("size", 0)) < 5_000_000  # skip huge files
                ]
                if not txt_files:
                    continue

                chosen = random.choice(txt_files)
                self._rl.wait()
                txt_url = f"https://archive.org/download/{identifier}/{chosen['name']}"
                text = _fetch(self._session, txt_url)

                if text and len(text) > 500:
                    yield title, text

            except Exception as e:
                logger.debug(f"[Archive.org] {e}")
                time.sleep(3)


class RSSCrawler:
    """
    Fetches articles from public RSS/Atom feeds.
    Follows links in feed entries and extracts article text.
    """

    NAME = "rss"
    DEFAULT_RATE = 1.5

    DEFAULT_FEEDS = [
        "https://feeds.bbci.co.uk/news/rss.xml",
        "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml",
        "https://www.theguardian.com/world/rss",
        "https://feeds.arstechnica.com/arstechnica/index",
        "https://www.scientificamerican.com/feed/rss/",
        "https://www.newscientist.com/feed/home/",
        "https://rss.slashdot.org/Slashdot/slashdotMain",
        "https://hnrss.org/frontpage",
    ]

    def __init__(self, feeds: list[str] = None, rate_limit: float = DEFAULT_RATE):
        self._rl = RateLimiter(rate_limit)
        self._session = _make_session()
        self._feeds = feeds or self.DEFAULT_FEEDS
        self._entry_queue: list[str] = []  # URLs to visit

    def stream(self) -> Iterator[tuple[str, str]]:
        while True:
            # Refill entry queue from feeds when empty
            if not self._entry_queue:
                self._refill()

            if not self._entry_queue:
                time.sleep(10)
                continue

            url = self._entry_queue.pop(0)
            try:
                self._rl.wait()
                html = _fetch(self._session, url)
                if not html:
                    continue
                text = CommonCrawlCrawler._html_to_text(html)
                if text and len(text) > 300:
                    yield url, text
            except Exception as e:
                logger.debug(f"[RSS] article fetch error: {e}")

    def _refill(self):
        """Parse all feeds and collect article URLs."""
        try:
            import xml.etree.ElementTree as ET
        except ImportError:
            return

        for feed_url in self._feeds:
            try:
                self._rl.wait()
                raw = _fetch(self._session, feed_url)
                if not raw:
                    continue

                root = ET.fromstring(raw)
                ns = {"atom": "http://www.w3.org/2005/Atom"}

                # RSS <link> tags
                for item in root.iter("item"):
                    link = item.findtext("link")
                    if link:
                        self._entry_queue.append(link.strip())

                # Atom <entry><link href=...>
                for entry in root.iter("{http://www.w3.org/2005/Atom}entry"):
                    link_el = entry.find("{http://www.w3.org/2005/Atom}link")
                    if link_el is not None:
                        href = link_el.get("href", "")
                        if href:
                            self._entry_queue.append(href)

            except Exception as e:
                logger.debug(f"[RSS] feed parse error {feed_url}: {e}")

        random.shuffle(self._entry_queue)
        logger.info(f"[RSS] Refilled queue: {len(self._entry_queue)} articles")


# ── Multi-source parallel crawler ─────────────────────────────────────

# Map source name → class
SOURCE_REGISTRY = {
    "wikipedia":    WikipediaCrawler,
    "gutenberg":    GutenbergCrawler,
    "commoncrawl":  CommonCrawlCrawler,
    "archive_org":  ArchiveOrgCrawler,
    "rss":          RSSCrawler,
}


@dataclass
class CrawlerConfig:
    sources: list[str] = field(
        default_factory=lambda: ["wikipedia", "gutenberg", "archive_org", "rss"]
    )
    # Per-source rate limits (seconds between requests)
    rate_limits: dict[str, float] = field(default_factory=dict)
    # How many docs to buffer ahead of training
    buffer_size: int = 200
    # Custom RSS feeds (only used if "rss" is in sources)
    rss_feeds: list[str] = field(default_factory=list)


class MultiCrawler:
    """
    Runs one daemon thread per source.  All threads push (title, text) into
    a shared queue.  The continuous trainer calls `drain(n)` to get the next
    batch of documents — this returns immediately if the buffer has enough
    data, or blocks briefly until it does.

    Because training takes minutes per epoch, the crawlers have plenty of
    time to pre-fill the buffer before the next round needs data.
    """

    def __init__(self, config: CrawlerConfig):
        self.config = config
        self._queue: queue.Queue = queue.Queue(maxsize=config.buffer_size)
        self._threads: list[threading.Thread] = []
        self._stop_event = threading.Event()
        self._stats: dict[str, int] = {s: 0 for s in config.sources}
        self._stats_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self):
        """Spawn one daemon thread per source."""
        for source_name in self.config.sources:
            cls = SOURCE_REGISTRY.get(source_name)
            if cls is None:
                logger.warning(f"Unknown crawler source: {source_name!r}. Skipping.")
                continue

            rate = self.config.rate_limits.get(source_name, cls.DEFAULT_RATE)
            kwargs = {"rate_limit": rate}
            if source_name == "rss" and self.config.rss_feeds:
                kwargs["feeds"] = self.config.rss_feeds

            crawler_instance = cls(**kwargs)
            t = threading.Thread(
                target=self._worker,
                args=(source_name, crawler_instance),
                daemon=True,
                name=f"crawler-{source_name}",
            )
            t.start()
            self._threads.append(t)
            logger.info(f"[MultiCrawler] Started {source_name} crawler (rate={rate}s)")

    def stop(self):
        """Signal all crawler threads to stop."""
        self._stop_event.set()

    def drain(self, n: int, timeout: float = 60.0) -> list[tuple[str, str]]:
        """
        Return up to `n` (title, text) pairs from the buffer.
        Waits up to `timeout` seconds for at least 1 item if the buffer
        is currently empty.
        """
        results = []
        deadline = time.time() + timeout

        while len(results) < n and time.time() < deadline:
            try:
                item = self._queue.get(timeout=1.0)
                results.append(item)
            except queue.Empty:
                if self._stop_event.is_set():
                    break

        return results

    def buffer_size(self) -> int:
        return self._queue.qsize()

    def stats(self) -> dict[str, int]:
        with self._stats_lock:
            return dict(self._stats)

    # ------------------------------------------------------------------
    # Worker
    # ------------------------------------------------------------------

    def _worker(self, source_name: str, crawler):
        """Runs in a daemon thread — feeds the shared queue."""
        logger.info(f"[{source_name}] Worker started")
        try:
            for title, text in crawler.stream():
                if self._stop_event.is_set():
                    break
                try:
                    # Block if queue is full (back-pressure)
                    self._queue.put((title, text), timeout=5.0)
                    with self._stats_lock:
                        self._stats[source_name] += 1
                    logger.debug(
                        f"[{source_name}] queued: {title[:50]} "
                        f"(buf={self._queue.qsize()})"
                    )
                except queue.Full:
                    # Buffer full — training hasn't consumed yet, just wait
                    time.sleep(1.0)
        except Exception as e:
            logger.error(f"[{source_name}] Worker crashed: {e}", exc_info=True)
        logger.info(f"[{source_name}] Worker stopped")
