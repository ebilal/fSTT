#!/usr/bin/env bash
set -euo pipefail

PY=${PY:-python}

$PY -m pytest -q
$PY scripts/train.py --dataset multiwoz --max_dialogs 50 --epochs 1 --device auto --build_index true --history_turns 4
$PY -m src.demo_offline --run outputs/latest --topk 5
