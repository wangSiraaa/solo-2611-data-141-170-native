"""Command line entry point.

The questionnaire is validated *before* any answer file is opened, so an
illegal (e.g. cyclic) graph is rejected without ever reading responses.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

from .engine import Engine
from .loader import load_inputs
from .questionnaire import QuestionnaireValidationError, load_questionnaire
from .report import write_outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Explain per-member question eligibility from a frozen questionnaire and "
            "existing responses. Answers are never imputed; the questionnaire is never edited."
        )
    )
    parser.add_argument("--questionnaire", required=True, help="path to questionnaire JSON")
    parser.add_argument("--households", required=True, help="path to households JSONL")
    parser.add_argument("--roster", required=True, help="path to roster JSONL")
    parser.add_argument("--responses", required=True, help="path to responses JSONL")
    parser.add_argument("--output-dir", required=True, help="directory for generated outputs")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    # Step 1: validate structure only.  No answer file is touched yet.
    try:
        questionnaire = load_questionnaire(args.questionnaire)
    except QuestionnaireValidationError as exc:
        payload = {
            "status": "rejected",
            "stage": "questionnaire_validation",
            "questionnaire": args.questionnaire,
            "problems": exc.problems,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2
    except (OSError, json.JSONDecodeError) as exc:
        print(
            json.dumps(
                {
                    "status": "rejected",
                    "stage": "questionnaire_load",
                    "questionnaire": args.questionnaire,
                    "problems": [str(exc)],
                },
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stderr,
        )
        return 2

    # Step 2: only now read the data files.
    data = load_inputs(args.households, args.roster, args.responses, questionnaire)
    engine = Engine(questionnaire, data)
    evaluation = engine.evaluate()
    paths = write_outputs(args.output_dir, questionnaire, data, evaluation, engine)

    print(
        json.dumps(
            {
                "status": "ok",
                "questionnaire_version": questionnaire.version,
                "instances_evaluated": len(evaluation.members),
                "issues": len(data.issues),
                "outputs": paths,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())