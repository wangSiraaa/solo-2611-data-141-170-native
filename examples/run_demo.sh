#!/bin/sh
# Reproduce the sample, edge-case and cyclic-rejection runs.
set -eu

cd "$(dirname "$0")/.."

export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

echo "== sample =="
python3 -m survey_eligibility \
  --questionnaire fixtures/questionnaire.json \
  --households    fixtures/households.jsonl \
  --roster        fixtures/roster.jsonl \
  --responses     fixtures/responses.jsonl \
  --output-dir    out/sample

echo "== edge cases =="
python3 -m survey_eligibility \
  --questionnaire fixtures/questionnaire.json \
  --households    examples/edge-cases/households.jsonl \
  --roster        examples/edge-cases/roster.jsonl \
  --responses     examples/edge-cases/responses.jsonl \
  --output-dir    out/edge

echo "== cyclic questionnaire (expected rejection, exit 2) =="
if python3 -m survey_eligibility \
  --questionnaire fixtures/cases/cyclic-questionnaire.json \
  --households    fixtures/households.jsonl \
  --roster        fixtures/roster.jsonl \
  --responses     fixtures/responses.jsonl \
  --output-dir    out/cyclic; then
  echo "ERROR: cyclic questionnaire was accepted" >&2
  exit 1
else
  echo "rejected as expected"
fi
