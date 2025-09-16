#!/usr/bin/env bash
set -e
python booru_posts_pipeline.py export-manifest --out data/manifest.csv
python gaudi/train_align.py --manifest data/manifest.csv --tag-desc data/tag_descriptions.json --out runs/align_s1 --epochs 2 --batch-size 32
python gaudi/pseudolabels.py --manifest data/manifest.csv --tag-desc data/tag_descriptions.json --align-ckpt runs/align_s1/checkpoints/last.pt --out runs/pseudomasks_v1
python gaudi/train_segmenter.py --manifest data/manifest.csv --pseudomasks runs/pseudomasks_v1 --align-out runs/align_s1 --out runs/seg_s1 --epochs 2
