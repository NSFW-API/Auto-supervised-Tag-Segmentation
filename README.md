# GaUDI — General-tag, Auto-supervised, Description-Informed (Tag→Mask)

This repo provides a *fully-automatic* training pipeline that learns **tag → segmentation mask(s)**
for Danbooru-style posts **without any human-drawn masks**. It only uses:
- Images with their **general** tags (from the Danbooru API), and
- A dictionary `tag → natural-language description(s)` that *you* provide or sync from the wiki.

The pipeline has two training stages:
1) **Image–Text Alignment** (multi-label CLIP-style) using only image-level general tags and your descriptions.
2) **Text-conditioned Segmentation** trained on **self-generated pseudo-masks** (no manual annotation).

It also includes an **unsupervised regionizer** (SLIC/felzenszwalb) to propose segments and
automatic validators (consistency & causal tests) to keep pseudo-labels clean.

> Designed to be practical to run on a single GPU; you can scale up later.

---

## Quickstart

### 0) Install deps
```bash
python -m pip install -r requirements.txt
```

### 1) Sync posts (images + general tags) & download media

> You can use your existing tag/wiki sync. This step fetches posts and saves images locally.

```bash
# ENV (set your Danbooru creds; basic auth strongly recommended)
export DANBOORU_USERNAME="your_username"
export DANBOORU_API_KEY="your_api_key"
export DANBOORU_USER_AGENT="YourCompany-GaUDI/1.0 (+you@example.com)"

# Initialize DB
python booru_posts_pipeline.py init-db

# Sync recent N posts (you can run multiple times; it resumes automatically)
python booru_posts_pipeline.py sync --limit 10000 --rating all --tags ""

# Download images (multi-threaded, resumable)
python booru_posts_pipeline.py fetch-images --max-workers 8 --skip-existing
```

**Notes**
- Only **general** tags are stored in the `post_tags` table (derived from `tag_string_general`).
- Images are saved under `data/images/xx/yy/md5.ext` with DB backrefs.

### 2) Build a dataset manifest & a tag description file
Export a CSV manifest and a JSON `tag_descriptions.json` (use your own dictionary or your tag pipeline).

```bash
python booru_posts_pipeline.py export-manifest --out data/manifest.csv
# Place your JSON mapping here (example below)
cat > data/tag_descriptions.json <<'JSON'
{
  "sky": ["the sky or heavens outdoors"],
  "ocean": ["sea or ocean water in the background"],
  "beach": ["sand near waterline"],
  "bikini": ["two-piece swimsuit worn by a person"],
  "shorts": ["short pants worn on the lower body"]
}
JSON
```

### 3) Stage A — Train the image–text alignment
```bash
python gaudi/train_align.py   --manifest data/manifest.csv   --tag-desc data/tag_descriptions.json   --out runs/align_s1   --epochs 2 --batch-size 32
```

### 4) Stage B — Generate pseudo-masks and train the segmenter (self-training)
```bash
# Round 1 — generate pseudo-masks
python gaudi/pseudolabels.py   --manifest data/manifest.csv   --tag-desc data/tag_descriptions.json   --align-ckpt runs/align_s1/checkpoints/last.pt   --out runs/pseudomasks_v1

# Train the segmenter on v1
python gaudi/train_segmenter.py   --manifest data/manifest.csv   --tag-desc data/tag_descriptions.json   --pseudomasks runs/pseudomasks_v1   --out runs/seg_s1 --epochs 2

# (Optional) Round 2 — refresh pseudo-masks using the trained segmenter
python gaudi/pseudolabels.py   --manifest data/manifest.csv   --tag-desc data/tag_descriptions.json   --align-ckpt runs/align_s1/checkpoints/last.pt   --segmenter-ckpt runs/seg_s1/checkpoints/last.pt   --out runs/pseudomasks_v2

python gaudi/train_segmenter.py   --manifest data/manifest.csv   --tag-desc data/tag_descriptions.json   --pseudomasks runs/pseudomasks_v2   --out runs/seg_s2 --epochs 2
```

### 5) Inference (get tag→mask for a new image)
```bash
python gaudi/infer.py   --image path/to/image.jpg   --tags "sky ocean beach bikini shorts"   --align-ckpt runs/align_s1/checkpoints/last.pt   --segmenter-ckpt runs/seg_s2/checkpoints/last.pt   --out out.json --viz out.png
```

---

## Files & folders
```
gaudi/
  data.py             # dataset & dataloaders
  models/
    image_text_model.py
    segmenter.py
  regionizer.py       # SLIC/felzenszwalb proposals
  pseudolabels.py     # Text-CAM + region scoring + validators
  train_align.py      # Stage A trainer (alignment)
  train_segmenter.py  # Stage B trainer (segmenter)
  infer.py            # Inference entry point
booru_posts_pipeline.py  # Posts sync + downloader + manifest export
data/
  images/             # downloaded images
  manifest.csv        # image path + general tags
  tag_descriptions.json
runs/
  ...                 # logs, TensorBoard, checkpoints
```

> This is a minimal, *working* scaffold. You can scale models, add stronger
> regionizers, and plug in CLIP/ViT backbones later.


---

## Tag → Tokens (LLM mapping)

If you have a *wiki-enriched tag JSON* (like your example), you can build a mapping from each **General** tag to a small list of **SentencePiece tokens** chosen via **OpenAI Structured Outputs**.

```bash
# 1) Ensure OPENAI_API_KEY is set
export OPENAI_API_KEY=sk-...

# 2) Run the builder
python gaudi/tag_tokens_builder.py   --tags-json path/to/your_tags.json   --spm-vocab path/to/spm.vocab.txt   --out-dir runs/tag_tokens_v1   --model gpt-4o-2024-08-06   --limit 6000 --concurrency 4
```

Artifacts:
- `runs/tag_tokens_v1/results.jsonl` — one record per tag: `{tag, tokens, token_ids, ...}`
- `runs/tag_tokens_v1/done.txt` — processed tag names (for resume)
- `runs/tag_tokens_v1/errors.jsonl` — failures (with messages)

This step **filters to General tags with a wiki**, sorts by `post_count` (desc), and uses a **candidate shortlist** per tag
(derived from the vocab + tag/aliases/wiki text) so the model only selects from valid tokens.


## Tag → Tokens with **File Search** (Responses API)

Upload your `spm.vocab.txt` to an OpenAI **vector store**, then let the model search it when selecting tokens.

```bash
export OPENAI_API_KEY=sk-...

python gaudi/tag_tokens_builder_fs.py   --tags-json path/to/your_tags.json   --spm-vocab path/to/spm.vocab.txt   --out-dir runs/tag_tokens_vfs_v1   --model gpt-4o-2024-08-06   --limit 6000 --concurrency 3
```

- The script **creates a vector store** (or reuses one) and uploads the vocab file (`files.create(...)` then `vector_stores.files.create(...)`).
- Each tag call uses **Responses API + tools=[{{"type":"file_search"}}]** so the model can search the full vocab.
- Results are **resumable** (`done.txt`) and mapped back to **token_ids** locally.
