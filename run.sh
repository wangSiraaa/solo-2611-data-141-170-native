#!/usr/bin/env bash
# 一键复现：两套输入的 fit/verify + apply 演示 + 随机压力核验。
set -euo pipefail
cd "$(dirname "$0")"

echo "== 主输入：fit =="
python3 -m rulelearn fit --input-dir fixtures --output-dir out/main

echo "== 主输入：apply 重新分流探针 =="
python3 -m rulelearn apply --rules out/main/rules.json --input-csv fixtures/probes.csv

echo "== 主输入：verify =="
python3 -m rulelearn verify --input-dir fixtures --output-dir out/main

echo "== 歧义微型输入：fit + verify =="
python3 -m rulelearn fit --input-dir cases/ambiguity --output-dir out/ambiguity
python3 -m rulelearn verify --input-dir cases/ambiguity --output-dir out/ambiguity

echo "== 随机压力核验（DP 对全穷举，300 组）=="
python3 tests/stress_vs_bruteforce.py

echo "全部完成，产物见 out/main 与 out/ambiguity。"
