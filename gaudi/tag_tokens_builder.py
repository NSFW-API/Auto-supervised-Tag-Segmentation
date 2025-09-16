#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tag_tokens_builder.py — Map Danbooru General tags (with wiki) → a small list of SentencePiece tokens
using OpenAI Structured Outputs. Prioritizes by post_count and is resumable.

Usage
-----
export OPENAI_API_KEY=sk-...                           # required
python tag_tokens_builder.py \
  --tags-json path/to/tags_with_wiki.json \
  --spm-vocab path/to/spm.vocab.txt \
  --out-dir runs/tag_tokens_v1 \
  --model gpt-4o-2024-08-06 \
  --limit 6000 \
  --concurrency 4

Outputs
-------
out-dir/
  results.jsonl        # one JSON object per tag with tokens & ids
  done.txt             # processed tag names (for resume)
  errors.jsonl         # failures
  tag_vocab.json       # ordered tag list actually processed
  config.json          # args snapshot

Notes
-----
- We do NOT upload the entire vocab to the LLM. We create a *candidate shortlist* per tag
  (up to ~300 tokens) via lexical matching against the tag name, aliases, and wiki signals.
- The LLM must choose ONLY from that shortlist; we enforce and correct if needed.
- spm.vocab.txt format supported: either one token per line, or "token\\tfreq".
"""
from __future__ import annotations

import argparse
import os
import json
import re
import time
import concurrent.futures as futures
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass

from openai import OpenAI, APIError, RateLimitError

# ----------------- I/O helpers -----------------
def load_tags_json(path: str) -> Dict[str, dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Accept either {"tags": {...}} or a raw dict
    if "tags" in data and isinstance(data["tags"], dict):
        return data["tags"]
    return data

def load_spm_vocab(path: str) -> Tuple[List[str], Dict[str,int]]:
    toks: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line: continue
            if "\\t" in line:
                # if the file literally contains backslash-t sequences (rare), handle both
                parts = line.split("\\t")
                tok = parts[0]
            elif "\t" in line:
                tok = line.split("\t")[0]
            else:
                tok = line
            toks.append(tok)
    token2id = {t:i for i,t in enumerate(toks)}
    return toks, token2id

def clean_text(s: str) -> str:
    s = s or ""
    s = s.replace("\\r", "\\n")
    # strip Danbooru dtext link markup [[...]]
    s = re.sub(r"\\[\\[([^\\]|#]+)(?:#[^\\]]*)?(?:\\|[^\\]]*)?\\]\\]", r"\\1", s)
    s = re.sub(r"\\{\\{[^}]+\\}\\}", " ", s)  # templates
    s = re.sub(r"\\s+", " ", s)
    return s.strip()

def normalize_candidates(words: List[str]) -> List[str]:
    out = []
    for w in words:
        w = w.strip().lower().replace(" ", "_")
        if w and w not in out:
            out.append(w)
    return out

# ----------------- Candidate shortlist -----------------
def shortlist_for_tag(tag: dict, vocab: List[str], token2id: Dict[str,int], max_len: int = 300) -> List[str]:
    """
    Build a candidate shortlist of vocab tokens likely relevant to this tag.
    Heuristics: direct forms, aliases, stems, plural/singular, underscore/▁ variants, in-wiki mentions.
    """
    name = tag["name"]
    aliases = tag.get("aliases") or []
    implied_by = tag.get("implied_by") or []
    wiki_body = ""
    if isinstance(tag.get("wiki"), dict):
        wiki_body = tag["wiki"].get("body") or ""
    wiki_body = clean_text(wiki_body)

    # base strings to search
    seeds = normalize_candidates([name] + aliases + implied_by)
    # harvest wiki content words (topical nouns, simple heuristic)
    words = re.findall(r"[A-Za-z][A-Za-z_\\-']{2,}", wiki_body.lower().replace(" ", "_"))
    seeds += normalize_candidates(words[:200])  # cap

    seeds = list(dict.fromkeys(seeds))  # preserve order, deduplicate

    # Generate forms
    forms = set()
    for s in seeds:
        forms.add(s)
        forms.add(s.replace("_", ""))
        if not s.endswith("s"): forms.add(s+"s")
        if s.endswith("s"): forms.add(s[:-1])
        forms.add(s.replace("_", "▁"))
        forms.add("▁"+s.replace("_", " "))
        forms.add("▁"+s.replace("_", ""))
        # try hyphen variants
        forms.add(s.replace("_","-"))
        forms.add(s.replace("-","_"))

    # Match vocab entries that contain any form (case-sensitive; SentencePiece tokens are case sensitive)
    cand = []
    for tok in vocab:
        flat = tok.lower()
        for f in forms:
            ff = f.lower()
            if ff and ff in flat:
                cand.append(tok)
                break
        if len(cand) >= max_len:
            break

    # Always include exacts if present
    for extra in [name, "▁"+name.replace("_"," "), "▁"+name.replace("_","")]:
        if extra in token2id and extra not in cand:
            cand.insert(0, extra)

    # De-dup preserve order
    seen=set(); out=[]
    for t in cand:
        if t not in seen:
            out.append(t); seen.add(t)
    return out[:max_len]

# ----------------- OpenAI call -----------------
SYSTEM_PROMPT = """You are selecting SentencePiece tokens to represent a Danbooru 'general' tag.
You will receive:
- The tag name, aliases, related tags, and its wiki body (plain text).
- A CANDIDATE LIST of vocabulary tokens (from spm.vocab.txt). You MUST select ONLY from that list.
Goal: choose 3-12 tokens that, *as a set*, best represent this tag's meaning in an embedding space.

Rules:
1) Only return tokens from the provided candidates. Do not invent tokens.
2) Prefer tokens that are semantically specific to the tag and common across usages.
3) Use '▁' word-start pieces when appropriate (it often denotes a word boundary).
4) Include synonyms if present in candidates (e.g., '▁breasts', '▁boobs').
5) Avoid tokens that primarily represent negations or exclusions from the wiki notes (e.g., 'flat chest' for 'breasts').
6) Order tokens from most to least representative.
7) If you believe a crucial concept is missing from the candidate list, add it to 'oov_suggestions' as plain strings (max 3), but do not include them in 'tokens'.
"""

def build_schema():
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "tag_token_selection",
            "schema": {
                "type": "object",
                "properties": {
                    "tokens": {
                        "type": "array",
                        "minItems": 3,
                        "maxItems": 12,
                        "items": {"type": "string"}
                    },
                    "oov_suggestions": {
                        "type": "array",
                        "maxItems": 3,
                        "items": {"type": "string"}
                    },
                    "notes": {"type": "string"}
                },
                "required": ["tokens"],
                "additionalProperties": False
            },
            "strict": True
        }
    }

def call_openai_select(client: OpenAI, model: str, tagobj: dict, candidates: List[str], timeout: int = 60) -> dict:
    name = tagobj["name"]
    aliases = tagobj.get("aliases") or []
    implied_by = tagobj.get("implied_by") or []
    groups = tagobj.get("groups") or []
    wiki_body = ""
    if isinstance(tagobj.get("wiki"), dict):
        wiki_body = tagobj["wiki"].get("body") or ""

    user_content = f"""
TAG: {name}
ALIASES: {', '.join(aliases[:20])}
RELATED: {', '.join(implied_by[:30])}
GROUPS: {', '.join(groups[:10])}

WIKI:
{clean_text(wiki_body)}

CANDIDATE_VOCAB_TOKENS (choose only from these; 1 per line):
{os.linesep.join(candidates)}
"""
    for attempt in range(6):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.1,
                response_format=build_schema()
            )
            msg = resp.choices[0].message
            if getattr(msg, "refusal", None):
                raise RuntimeError(f"Model refusal: {msg.refusal}")
            content = msg.content
            data = json.loads(content)
            return data
        except RateLimitError as e:
            time.sleep(2**attempt + 0.1)
        except APIError as e:
            time.sleep(1.5**attempt + 0.1)
        except Exception as e:
            if attempt >= 5:
                raise
            time.sleep(1.2**attempt + 0.1)
    raise RuntimeError("exhausted retries")

# ----------------- Main -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags-json", required=True, help="Path to your tag JSON (as shown in user example)")
    ap.add_argument("--spm-vocab", required=True, help="SentencePiece vocab text file")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", default=os.getenv("GAUDI_OPENAI_MODEL", "gpt-4o-2024-08-06"))
    ap.add_argument("--limit", type=int, default=1000000)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--min-posts", type=int, default=1)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    tags = load_tags_json(args.tags_json)
    vocab, token2id = load_spm_vocab(args.spm_vocab)

    # Filter: General + has wiki
    items = []
    for name, obj in tags.items():
        cat = obj.get("category") or obj.get("category_name") or ""
        if str(cat).lower() not in ("0", "general", "category 0"):
            continue
        if not obj.get("wiki"):
            continue
        if obj.get("post_count", 0) < args.min_posts:
            continue
        items.append(obj)
    # Sort by post_count desc
    items.sort(key=lambda x: int(x.get("post_count", 0)), reverse=True)
    items = items[:args.limit]

    # Resume: read done set
    done_path = os.path.join(args.out_dir, "done.txt")
    done = set()
    if os.path.exists(done_path):
        with open(done_path, "r", encoding="utf-8") as f:
            done = set([ln.strip() for ln in f if ln.strip()])
    results_path = os.path.join(args.out_dir, "results.jsonl")
    errors_path = os.path.join(args.out_dir, "errors.jsonl")

    # Save the ordered tag list for reproducibility
    with open(os.path.join(args.out_dir, "tag_vocab.json"), "w", encoding="utf-8") as f:
        json.dump([it["name"] for it in items], f, indent=2)

    client = OpenAI()

    def process_one(tagobj: dict):
        name = tagobj["name"]
        cand = shortlist_for_tag(tagobj, vocab, token2id)
        data = call_openai_select(client, args.model, tagobj, cand)
        # enforce correctness: keep only tokens from cand, dedup, map to ids
        tokens = []
        for t in data.get("tokens", []):
            if t in cand and t not in tokens:
                tokens.append(t)
        # map to ids
        token_ids = [token2id[t] for t in tokens if t in token2id]
        out = {
            "tag": name,
            "tokens": tokens,
            "token_ids": token_ids,
            "notes": data.get("notes",""),
            "oov_suggestions": data.get("oov_suggestions", []),
            "post_count": tagobj.get("post_count", 0),
        }
        return out

    # Thread pool with resume-safe writes
    ok = 0
    with futures.ThreadPoolExecutor(max_workers=args.concurrency) as ex, \
         open(results_path, "a", encoding="utf-8") as fout, \
         open(errors_path, "a", encoding="utf-8") as ferr:

        futs = {}
        for tagobj in items:
            name = tagobj["name"]
            if name in done:
                continue
            futs[ex.submit(process_one, tagobj)] = name

        for fut in futures.as_completed(futs):
            name = futs[fut]
            try:
                rec = fut.result()
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                with open(done_path, "a", encoding="utf-8") as fdone:
                    fdone.write(name + "\n")
                ok += 1
                if ok % 25 == 0:
                    print(f"[progress] completed {ok}")
            except Exception as e:
                ferr.write(json.dumps({"tag": name, "error": str(e)}) + "\n")
                ferr.flush()

    print(f"Done. Wrote {ok} records to {results_path}")

if __name__ == "__main__":
    main()
