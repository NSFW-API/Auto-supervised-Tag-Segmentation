#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tag_tokens_builder_fs.py — Same as tag_tokens_builder.py, but uses **OpenAI File Search** via the Responses API
so the model can search your *entire* SentencePiece vocab file in a hosted vector store.

Pipeline:
1) Ensure a vector store exists (create if missing).
2) Upload (or re-use) your spm.vocab.txt into that vector store.
3) Iterate over **General** tags with wiki (sorted by post_count desc).
4) For each tag, call Responses API with tools=[{"type":"file_search", "vector_store_ids":[...]}]
   and Structured Outputs schema to return 3–12 tokens strictly present in the vocab file.
5) Verify tokens exist locally, map to token_ids, write JSONL results (resumable).

Docs
----
- Responses API + file_search tool (official cookbook example): https://cookbook.openai.com/examples/file_search_responses
- Vector stores (create store; attach files): https://platform.openai.com/docs/api-reference/vector-stores-files/createFile

Usage
-----
export OPENAI_API_KEY=sk-...
python gaudi/tag_tokens_builder_fs.py \
  --tags-json path/to/your_tags.json \
  --spm-vocab path/to/spm.vocab.txt \
  --out-dir runs/tag_tokens_vfs_v1 \
  --model gpt-4o-2024-08-06 \
  --limit 6000 \
  --concurrency 3

Resumes from out-dir/done.txt.
"""
from __future__ import annotations

import argparse, os, json, re, time, random
import concurrent.futures as futures
from typing import Dict, List, Tuple, Any

from openai import OpenAI, APIError, RateLimitError

# --------------- helpers ---------------
def load_tags_json(path: str) -> Dict[str, dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["tags"] if isinstance(data, dict) and "tags" in data else data

def load_spm_vocab(path: str):
    toks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line: continue
            toks.append(line.split("\t")[0] if "\t" in line else line)
    token2id = {t:i for i,t in enumerate(toks)}
    return toks, token2id

def clean_dtext(s: str) -> str:
    s = (s or "").replace("\r","\n")
    s = re.sub(r"\[\[([^\]|#]+)(?:#[^\]]*)?(?:\|[^\]]*)?\]\]", r"\1", s)
    s = re.sub(r"\s+"," ", s)
    return s.strip()

# --------------- vector store ---------------
def ensure_vector_store(client: OpenAI, out_dir: str, vocab_path: str, name: str = "gaudi_spm_vocab") -> str:
    os.makedirs(out_dir, exist_ok=True)
    meta = os.path.join(out_dir, "vector_store.json")
    if os.path.exists(meta):
        try:
            vs = json.load(open(meta,"r",encoding="utf-8"))
            if vs.get("id"):
                return vs["id"]
        except Exception:
            pass
    # create, upload, attach
    vs = client.vector_stores.create(name=name)
    file_obj = client.files.create(file=open(vocab_path,"rb"), purpose="assistants")
    client.vector_stores.files.create(vector_store_id=vs.id, file_id=file_obj.id)
    with open(meta,"w",encoding="utf-8") as f:
        json.dump({"id": vs.id, "file_id": file_obj.id}, f, indent=2)
    return vs.id

# --------------- LLM call ---------------
SYSTEM_PROMPT = """You will select SentencePiece tokens to represent a Danbooru 'general' tag.
You have access to the *entire* SentencePiece vocabulary via the file_search tool.
Rules:
- Choose **3–12** tokens that *together* best represent the tag semantics.
- Choose tokens **ONLY if they exist in the vocabulary file**.
- Prefer word-start '▁' pieces for whole words; include common synonyms when available.
- Avoid tokens that mainly express exclusions in the wiki (e.g., 'flat chest' for 'breasts').
- Order tokens from most to least representative.
Return JSON that matches the provided strict schema.
"""

def build_schema():
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "tag_token_selection",
            "schema": {
                "type": "object",
                "properties": {
                    "tokens": {"type": "array", "items": {"type": "string"}, "minItems": 3, "maxItems": 12},
                    "oov_suggestions": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
                    "notes": {"type": "string"}
                },
                "required": ["tokens"],
                "additionalProperties": False
            },
            "strict": True
        }
    }

def call_with_file_search(client: OpenAI, model: str, vector_store_id: str, tag: dict) -> dict:
    name = tag["name"]
    aliases = tag.get("aliases") or []
    implied_by = tag.get("implied_by") or []
    groups = tag.get("groups") or []
    wiki = (tag.get("wiki") or {}).get("body","")

    # We instruct the model to *use file_search* to find tokens; we also give it search hints.
    # The actual vocab content is inside the vector store (spm.vocab.txt).
    user_prompt = f"""
TAG: {name}
ALIASES: {', '.join(aliases[:20])}
RELATED: {', '.join(implied_by[:30])}
GROUPS: {', '.join(groups[:10])}

WIKI (plain text):
{clean_dtext(wiki)}

INSTRUCTIONS:
- Use the file_search tool to scan the SentencePiece vocabulary (spm.vocab.txt) for relevant pieces (full-word '▁...' and subword variants).
- Return ONLY tokens that exist in the vocab file.
- If a crucial synonym isn't in vocab, add it to 'oov_suggestions' (plain text), but not in 'tokens'.
"""

    for attempt in range(6):
        try:
            resp = client.responses.create(
                model=model,
                input=user_prompt,
                tools=[{
                    "type":"file_search",
                    "vector_store_ids": [vector_store_id],
                    # you can also set "max_num_results": N, and metadata filters if you attach multiple files
                }],
                response_format=build_schema(),
                # include file search results to debug retrieval if desired:
                include=["file_search_call.results"]
            )
            # The model's structured output is in resp.output (message item)
            # Extract the assistant message content (json text) safely:
            # Some SDKs expose response.output_text, but we stick to parsing the message item.
            msg_items = [o for o in resp.output if o.get("type")=="message"]
            if not msg_items:
                raise RuntimeError("No assistant message in response")
            content_items = msg_items[0]["content"]
            txt_parts = [c["text"] for c in content_items if c["type"]=="output_text"]
            data = json.loads("".join(txt_parts))
            return data, resp
        except RateLimitError:
            time.sleep(2**attempt + 0.1)
        except APIError:
            time.sleep(1.5**attempt + 0.1)
        except Exception as e:
            if attempt >= 5:
                raise
            time.sleep(1.2**attempt + 0.1)

# --------------- main ---------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags-json", required=True)
    ap.add_argument("--spm-vocab", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", default=os.getenv("GAUDI_OPENAI_MODEL", "gpt-4o-2024-08-06"))
    ap.add_argument("--limit", type=int, default=1000000)
    ap.add_argument("--min-posts", type=int, default=1)
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument("--vector-store-id", default=None, help="If provided, re-use this store; otherwise create+upload")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    json.dump(vars(args), open(os.path.join(args.out_dir, "config.json"),"w"), indent=2)

    tags = load_tags_json(args.tags_json)
    vocab, token2id = load_spm_vocab(args.spm_vocab)

    # filter & sort
    items = []
    for name, obj in tags.items():
        cat = str(obj.get("category") or "").lower()
        if cat not in ("0","general","category 0"):
            continue
        if not obj.get("wiki"):
            continue
        if int(obj.get("post_count",0)) < args.min_posts:
            continue
        items.append(obj)
    items.sort(key=lambda x: int(x.get("post_count",0)), reverse=True)
    items = items[:args.limit]

    # vector store
    client = OpenAI()
    vs_id = args.vector_store_id or ensure_vector_store(client, args.out_dir, args.spm_vocab)

    # resume
    done_path = os.path.join(args.out_dir, "done.txt")
    done = set()
    if os.path.exists(done_path):
        done = set([ln.strip() for ln in open(done_path,"r",encoding="utf-8") if ln.strip()])
    res_path = os.path.join(args.out_dir, "results.jsonl")
    err_path = os.path.join(args.out_dir, "errors.jsonl")
    tv_path  = os.path.join(args.out_dir, "tag_vocab.json")
    json.dump([it["name"] for it in items], open(tv_path,"w",encoding="utf-8"), indent=2)

    ok = 0
    def process(tagobj):
        data, resp = call_with_file_search(client, args.model, vs_id, tagobj)
        # Enforce: tokens must exist in vocab
        toks = []
        for t in data.get("tokens", []):
            if t in token2id and t not in toks:
                toks.append(t)
        out = {
            "tag": tagobj["name"],
            "tokens": toks,
            "token_ids": [token2id[t] for t in toks],
            "oov_suggestions": data.get("oov_suggestions", []),
            "notes": data.get("notes",""),
            "post_count": tagobj.get("post_count",0),
            # Store the ids of files searched for debugging
            "file_search": {
                "vector_store_id": vs_id,
            }
        }
        return out

    with futures.ThreadPoolExecutor(max_workers=args.concurrency) as ex, \
         open(res_path,"a",encoding="utf-8") as fout, \
         open(err_path,"a",encoding="utf-8") as ferr:
        futs = {}
        for obj in items:
            if obj["name"] in done: 
                continue
            futs[ex.submit(process, obj)] = obj["name"]

        for fu in futures.as_completed(futs):
            name = futs[fu]
            try:
                rec = fu.result()
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n"); fout.flush()
                with open(done_path,"a",encoding="utf-8") as fdone:
                    fdone.write(name+"\n")
                ok += 1
                if ok % 25 == 0:
                    print(f"[progress] {ok} tags")
            except Exception as e:
                ferr.write(json.dumps({"tag":name,"error":str(e)}) + "\n"); ferr.flush()

    print(f"Done: wrote {ok} records to {res_path}")

if __name__ == "__main__":
    main()
