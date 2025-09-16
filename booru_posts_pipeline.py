#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Danbooru Posts Pipeline — sync posts (images) with general tags, download media,
and export a manifest suitable for GaUDI training.

- Polite rate limiting & retries
- Descending ID cursor to avoid deep paging 410s
- SQLite storage for posts + post_tags
- Multi-threaded image downloader with resume
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import re
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple
from urllib.parse import urlencode, urlparse
import pathlib

import requests
from requests.adapters import HTTPAdapter, Retry

DB_PATH   = os.getenv("DANBOORU_DB", "danbooru_posts.db")
BASE_URL  = os.getenv("DANBOORU_BASE", "https://danbooru.donmai.us")
USER_AGENT = os.getenv("DANBOORU_USER_AGENT", "GaUDI-Posts/1.0 (+contact)")
USERNAME   = os.getenv("DANBOORU_USERNAME") or os.getenv("DANBOORU_LOGIN") or os.getenv("DANBOORU_USER")
API_KEY    = os.getenv("DANBOORU_API_KEY")
RPS        = float(os.getenv("DANBOORU_RPS", "1.0"))
MIN_DELAY  = max(1.0 / max(RPS, 0.1), 0.2)

IMG_ROOT = os.getenv("GAUDI_IMG_ROOT", "data/images")

logger = logging.getLogger("gaudi.posts")

SCHEMA_SQL = r"""
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS meta (
  key TEXT PRIMARY KEY,
  value TEXT
);

CREATE TABLE IF NOT EXISTS posts (
  id INTEGER PRIMARY KEY,
  md5 TEXT,
  file_ext TEXT,
  image_width INTEGER,
  image_height INTEGER,
  file_size INTEGER,
  rating TEXT,
  score INTEGER,
  fav_count INTEGER,
  source TEXT,
  created_at TEXT,
  updated_at TEXT,
  file_url TEXT,
  large_file_url TEXT,
  preview_file_url TEXT,
  local_path TEXT,
  downloaded INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_posts_rating ON posts(rating);
CREATE INDEX IF NOT EXISTS idx_posts_md5 ON posts(md5);
CREATE INDEX IF NOT EXISTS idx_posts_downloaded ON posts(downloaded);

CREATE TABLE IF NOT EXISTS post_tags (
  post_id INTEGER NOT NULL,
  tag TEXT NOT NULL,
  PRIMARY KEY (post_id, tag),
  FOREIGN KEY(post_id) REFERENCES posts(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_post_tags_tag ON post_tags(tag);
"""

# --------------- DB helpers ---------------
def connect_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn

def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_SQL)
    conn.commit()

def set_state(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute("""INSERT INTO meta(key,value) VALUES(?,?)
                    ON CONFLICT(key) DO UPDATE SET value=excluded.value""", (key, value))
    conn.commit()

def get_state(conn: sqlite3.Connection, key: str, default: Optional[str] = None) -> Optional[str]:
    row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return row[0] if row else default

# --------------- HTTP client ---------------
@dataclass
class BooruSession:
    sess: requests.Session
    last_ts: float = 0.0

def build_session() -> BooruSession:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    if USERNAME and API_KEY:
        s.auth = (USERNAME, API_KEY)
    retries = Retry(
        total=8, connect=8, read=8, status=8,
        backoff_factor=1.25,
        status_forcelist=(408, 420, 429, 500, 502, 503, 504),
        allowed_methods=("GET", "HEAD"),
        raise_on_status=False,
        respect_retry_after_header=True,
    )
    s.mount("https://", HTTPAdapter(max_retries=retries, pool_maxsize=32, pool_connections=32))
    s.mount("http://",  HTTPAdapter(max_retries=retries, pool_maxsize=32, pool_connections=32))
    return BooruSession(sess=s)

def _throttle(b: BooruSession):
    dt = time.time() - b.last_ts
    if dt < MIN_DELAY:
        time.sleep(MIN_DELAY - dt)
    b.last_ts = time.time()

def get_json(b: BooruSession, path: str, params: Dict[str, Any]) -> Any:
    _throttle(b)
    url = f"{BASE_URL}{path}"
    q = dict(params)
    if not (USERNAME and API_KEY):
        if USERNAME: q["login"] = USERNAME
        if API_KEY:  q["api_key"] = API_KEY
    r = b.sess.get(url, params=q, timeout=(15, 90))
    if r.status_code == 410:
        return []
    r.raise_for_status()
    try:
        return r.json()
    except Exception as e:
        raise RuntimeError(f"Non-JSON response for {url}?{urlencode(q)}: {e}; body[:200]={r.text[:200]!r}")

def iter_cursor_desc(b: BooruSession, path: str, limit: int = 100, base_params: Optional[Dict[str, Any]] = None,
                     id_field: str = "id", first_cursor: Optional[int] = None) -> Iterator[List[Dict[str, Any]]]:
    params = dict(base_params or {})
    params.setdefault("search[order]", "id")  # desc by default
    cursor = first_cursor
    while True:
        q = dict(params)
        q["limit"] = limit
        if cursor is not None:
            q["search[id_lt]"] = cursor
        data = get_json(b, path, q)
        if not data:
            break
        yield data
        cursor = min(int(it.get(id_field, 0) or 0) for it in data)

# --------------- Sync posts ---------------
FIELDS = [
    "id","tag_string_general","md5","file_ext","image_width","image_height","file_size",
    "rating","score","fav_count","source","created_at","updated_at",
    "file_url","large_file_url","preview_file_url"
]

def normalize_post(row: Dict[str, Any]) -> Dict[str, Any]:
    d = {k: row.get(k) for k in FIELDS}
    # keep only general tags and split
    tags = [t for t in (d.get("tag_string_general") or "").split() if t]
    d["_general_tags"] = tags
    return d

def upsert_posts(conn: sqlite3.Connection, rows: List[Dict[str, Any]]) -> int:
    if not rows: return 0
    sql = """
    INSERT INTO posts (id, md5, file_ext, image_width, image_height, file_size, rating, score, fav_count,
                       source, created_at, updated_at, file_url, large_file_url, preview_file_url, local_path, downloaded)
    VALUES (:id, :md5, :file_ext, :image_width, :image_height, :file_size, :rating, :score, :fav_count,
            :source, :created_at, :updated_at, :file_url, :large_file_url, :preview_file_url, 
            COALESCE((SELECT local_path FROM posts WHERE id=:id), NULL),
            COALESCE((SELECT downloaded FROM posts WHERE id=:id), 0))
    ON CONFLICT(id) DO UPDATE SET
      md5=excluded.md5,
      file_ext=excluded.file_ext,
      image_width=excluded.image_width,
      image_height=excluded.image_height,
      file_size=excluded.file_size,
      rating=excluded.rating,
      score=excluded.score,
      fav_count=excluded.fav_count,
      source=excluded.source,
      created_at=excluded.created_at,
      updated_at=excluded.updated_at,
      file_url=excluded.file_url,
      large_file_url=excluded.large_file_url,
      preview_file_url=excluded.preview_file_url
    """
    conn.executemany(sql, rows)
    # post_tags
    tag_pairs = []
    for r in rows:
        pid = r["id"]
        for t in r["_general_tags"]:
            tag_pairs.append((pid, t))
    if tag_pairs:
        conn.executemany("INSERT OR IGNORE INTO post_tags (post_id, tag) VALUES (?,?)", tag_pairs)
    conn.commit()
    return len(rows)

def sync_posts(conn: sqlite3.Connection, b: BooruSession, limit: int = 10000,
               rating: str = "all", tags: str = "", only_images: bool = True) -> int:
    first_cursor_s = get_state(conn, "posts_cursor_desc_min", None)
    first_cursor = int(first_cursor_s) if first_cursor_s is not None else None
    logger.info("Syncing posts starting id_lt=%s ...", first_cursor)
    params = {
        "only": ",".join(FIELDS),
        "random": "false",
        "search[order]": "id"
    }
    if rating != "all":
        params["tags"] = f"rating:{rating} {tags}".strip()
    else:
        if tags:
            params["tags"] = tags
    total = 0
    for batch_idx, chunk in enumerate(iter_cursor_desc(b, "/posts.json", limit=100, base_params=params, first_cursor=first_cursor), start=1):
        rows = [normalize_post(x) for x in chunk]
        n = upsert_posts(conn, rows)
        total += n
        new_cursor = min(r["id"] for r in rows)
        set_state(conn, "posts_cursor_desc_min", str(new_cursor))
        logger.info("posts batch %d: +%d (total=%d, next id_lt <%d)", batch_idx, n, total, new_cursor)
        if total >= limit:
            logger.info("Reached sync limit=%d", limit)
            break
    return total

# --------------- Downloader ---------------
def _md5_path(md5: str, ext: str) -> str:
    ext = (ext or "jpg").strip(".")
    p = os.path.join(IMG_ROOT, md5[:2], md5[2:4])
    os.makedirs(p, exist_ok=True)
    return os.path.join(p, f"{md5}.{ext}")

def download_one(b: BooruSession, row: Tuple[int, str, str, str]) -> Tuple[int, str, bool, str]:
    pid, md5, ext, url = row
    if not url:
        return pid, "", False, "no_url"
    path = _md5_path(md5 or hashlib.md5(str(pid).encode()).hexdigest(), ext or "jpg")
    if os.path.exists(path):
        return pid, path, True, "exists"
    try:
        r = b.sess.get(url, stream=True, timeout=(15, 90))
        if r.status_code == 403:
            return pid, "", False, "403"
        r.raise_for_status()
        with open(path, "wb") as f:
            for ch in r.iter_content(chunk_size=1<<16):
                if ch: f.write(ch)
        return pid, path, True, "ok"
    except Exception as e:
        return pid, "", False, f"error:{e}"

def fetch_images(conn: sqlite3.Connection, b: BooruSession, max_workers: int = 8, skip_existing: bool = True, prefer_large: bool = True) -> Tuple[int,int]:
    rows = conn.execute("""SELECT id, md5, file_ext, COALESCE(large_file_url,file_url) AS url, local_path, downloaded
                           FROM posts ORDER BY id DESC""").fetchall()
    tasks = []
    ok = 0
    fail = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for pid, md5, ext, url, local, done in rows:
            if skip_existing and done and local and os.path.exists(local):
                continue
            tasks.append(ex.submit(download_one, b, (pid, md5, ext, url)))
        for fut in as_completed(tasks):
            pid, path, success, msg = fut.result()
            if success:
                conn.execute("UPDATE posts SET local_path=?, downloaded=1 WHERE id=?", (path, pid))
                ok += 1
            else:
                fail += 1
            if (ok+fail) % 100 == 0:
                conn.commit()
                logger.info("download progress ok=%d fail=%d", ok, fail)
    conn.commit()
    logger.info("downloads done ok=%d fail=%d", ok, fail)
    return ok, fail

# --------------- Manifest export ---------------
def export_manifest(conn: sqlite3.Connection, out_path: str = "data/manifest.csv") -> int:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cur = conn.execute("""SELECT p.id, p.local_path, GROUP_CONCAT(pt.tag, ' ') AS tags
                          FROM posts p LEFT JOIN post_tags pt ON pt.post_id=p.id
                          WHERE p.downloaded=1 AND p.local_path IS NOT NULL
                          GROUP BY p.id ORDER BY p.id DESC""")
    n = 0
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["post_id","image_path","tags"])
        for pid, path, tags in cur.fetchall():
            if not path or not os.path.exists(path):
                continue
            w.writerow([pid, path, tags or ""])
            n += 1
    logger.info("Exported manifest rows=%d → %s", n, out_path)
    return n

# --------------- CLI ---------------
def configure_logging(verbosity:int=1, logfile:str="gaudi_posts.log") -> None:
    level = logging.WARNING if verbosity <= 0 else (logging.INFO if verbosity == 1 else logging.DEBUG)
    fmt = "%(asctime)s %(levelname)s %(message)s"
    for h in list(logger.handlers):
        logger.removeHandler(h)
    logger.setLevel(level)
    ch = logging.StreamHandler(sys.stdout); ch.setLevel(level); ch.setFormatter(logging.Formatter(fmt))
    fh = logging.FileHandler(logfile, encoding="utf-8"); fh.setLevel(level); fh.setFormatter(logging.Formatter(fmt))
    logger.addHandler(ch); logger.addHandler(fh)

def main():
    p = argparse.ArgumentParser(description="Danbooru posts sync/downloader/manifest for GaUDI")
    p.add_argument("-v","--verbose", action="count", default=1)
    p.add_argument("-q","--quiet", action="store_true")
    p.add_argument("--log-file", default="gaudi_posts.log")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp0 = sub.add_parser("init-db"); sp0.set_defaults(func=lambda a: (ensure_schema(connect_db()), print(f"DB ready at {DB_PATH}")))
    sp1 = sub.add_parser("sync"); 
    sp1.add_argument("--limit", type=int, default=10000)
    sp1.add_argument("--rating", choices=["all","s","q","e"], default="all", help="s=safe, q=questionable, e=explicit")
    sp1.add_argument("--tags", default="", help="additional search terms; leave empty for all")
    def _do_sync(a):
        conn=connect_db(); ensure_schema(conn); b=build_session(); n=sync_posts(conn,b,limit=a.limit,rating=a.rating,tags=a.tags); print(f"synced {n}")
    sp1.set_defaults(func=_do_sync)

    sp2 = sub.add_parser("fetch-images")
    sp2.add_argument("--max-workers", type=int, default=8)
    sp2.add_argument("--skip-existing", action="store_true", default=True)
    def _do_fetch(a):
        conn=connect_db(); ensure_schema(conn); b=build_session(); ok,fail=fetch_images(conn,b,a.max_workers,a.skip_existing); print(f"ok={ok} fail={fail}")
    sp2.set_defaults(func=_do_fetch)

    sp3 = sub.add_parser("export-manifest")
    sp3.add_argument("--out", default="data/manifest.csv")
    def _do_export(a):
        conn=connect_db(); ensure_schema(conn); export_manifest(conn, a.out)
    sp3.set_defaults(func=_do_export)

    args = p.parse_args()
    verbosity = 0 if args.quiet else args.verbose
    configure_logging(verbosity=verbosity, logfile=args.log_file)
    args.func(args)

if __name__ == "__main__":
    main()
