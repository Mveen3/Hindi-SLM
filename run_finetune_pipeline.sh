#!/usr/bin/env bash
set -eo pipefail

PYTHON="/home/mveen/miniconda3/envs/ml/bin/python"
LOGFILE="finetuning_pipeline.log"

echo "==========================================================" | tee "$LOGFILE"
echo "Starting Optimized SLM Finetuning & Reasoning Evaluation Pipeline" | tee -a "$LOGFILE"
echo "Timestamp: $(date)" | tee -a "$LOGFILE"
echo "Python: $PYTHON" | tee -a "$LOGFILE"
echo "==========================================================" | tee -a "$LOGFILE"

echo "" | tee -a "$LOGFILE"
echo ">>> [1/4] Finetuning Hindi Model H (fresh optimized run)..." | tee -a "$LOGFILE"
$PYTHON -u main.py finetune --lang hindi --fresh 2>&1 | tee -a "$LOGFILE"

echo "" | tee -a "$LOGFILE"
echo ">>> [2/4] Finetuning Nepali Model L (fresh optimized run)..." | tee -a "$LOGFILE"
$PYTHON -u main.py finetune --lang nepali --fresh 2>&1 | tee -a "$LOGFILE"

echo "" | tee -a "$LOGFILE"
echo ">>> [3/4] Evaluating Reasoning on Hindi Model H (best checkpoint)..." | tee -a "$LOGFILE"
$PYTHON -u main.py evaluate-reasoning --lang hindi 2>&1 | tee -a "$LOGFILE"

echo "" | tee -a "$LOGFILE"
echo ">>> [4/4] Evaluating Reasoning on Nepali Model L (best checkpoint)..." | tee -a "$LOGFILE"
$PYTHON -u main.py evaluate-reasoning --lang nepali 2>&1 | tee -a "$LOGFILE"

echo "" | tee -a "$LOGFILE"
echo "==========================================================" | tee -a "$LOGFILE"
echo "Optimized Pipeline completed successfully!" | tee -a "$LOGFILE"
echo "Timestamp: $(date)" | tee -a "$LOGFILE"
echo "==========================================================" | tee -a "$LOGFILE"
