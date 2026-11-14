"""Command line interface: train / precheck / encode / decode."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .artifacts import ArtifactError, load_model, precheck, save_artifacts
from .corpus import (
    CorpusError,
    params_from_config,
    read_config,
    read_corpus,
    read_text_records,
)
from .encoder import BPECodec, DecodeError
from .trainer import IncrementalBPETrainer


def _cmd_train(args: argparse.Namespace) -> int:
    config = read_config(args.config)
    params = params_from_config(config, where=str(args.config))
    records = read_corpus(args.corpus)
    trainer = IncrementalBPETrainer(params).fit(records)
    summary = trainer.summary()
    artifact_config = {
        "merge_steps": params.merge_steps,
        "min_weighted_pair_count": params.min_weighted_pair_count,
        "descriptions": {
            key: value for key, value in config.items() if isinstance(value, str)
        },
    }
    save_artifacts(
        args.out,
        symbols=trainer.symbols.symbols,
        merges=trainer.merges,
        config=artifact_config,
        summary=summary,
    )
    print(f"trained on {summary['training_records']} records, "
          f"{summary['unique_segments']} unique segments")
    print(f"stop reason: {summary['stop_reason']} "
          f"after {summary['rounds_completed']} rounds; "
          f"vocab size {summary['vocab_size']}")
    print("incremental maintenance counters:")
    print(f"  positions checked (replacement walks): {summary['positions_checked']}")
    print(f"  adjacency updates:                     {summary['adjacency_updates']}")
    print(f"  stale candidates evicted:              {summary['candidates_evicted']}")
    print(f"  initial full scan positions:           {summary['initial_scan_positions']}")
    print(f"  naive per-round rescan equivalent:     {summary['full_rescan_positions_equivalent']}")
    print(f"artifacts written to {args.out}/ "
          f"(vocab.json, merges.json, config.json, summary.json)")
    return 0


def _cmd_precheck(args: argparse.Namespace) -> int:
    problems = precheck(args.artifacts)
    if problems:
        for problem in problems:
            print(f"error: {problem}", file=sys.stderr)
        return 1
    model = load_model(args.artifacts)
    print(f"ok: {len(model.symbols)} symbols, {len(model.merges)} merges, "
          f"stop reason {model.summary['stop_reason']}")
    return 0


def _iter_encode_records(args: argparse.Namespace):
    """Yield (record_index, raw_bytes) for the requested input mode."""
    if args.jsonl:
        for index, text in enumerate(read_text_records(args.jsonl)):
            yield index, text.encode("utf-8")
    else:
        yield 0, Path(args.input).read_bytes()


def _cmd_encode(args: argparse.Namespace) -> int:
    codec = BPECodec.from_artifacts(args.artifacts)
    out_handle = open(args.out, "w", encoding="utf-8") if args.out else sys.stdout
    try:
        for index, data in _iter_encode_records(args):
            try:
                tokens = codec.encode_bytes(data)
            except UnicodeDecodeError as exc:
                raise CorpusError(
                    f"record {index}: input is not valid UTF-8 ({exc})"
                ) from exc
            line = {
                "record": index,
                "byte_length": len(data),
                "tokens": [
                    {
                        "id": token.id,
                        "start": token.start,
                        "end": token.end,
                        "hex": codec.symbols[token.id].hex(),
                    }
                    for token in tokens
                ],
            }
            out_handle.write(json.dumps(line, ensure_ascii=False) + "\n")
    finally:
        if args.out:
            out_handle.close()
    return 0


def _decode_record(codec: BPECodec, obj: dict, where: str) -> bytes:
    if not isinstance(obj, dict) or not isinstance(obj.get("tokens"), list):
        raise DecodeError(f"{where}: expected an object with a 'tokens' list")
    byte_length = obj.get("byte_length")
    if isinstance(byte_length, bool) or not isinstance(byte_length, int) or byte_length < 0:
        raise DecodeError(f"{where}: 'byte_length' must be a non-negative integer")
    ids = []
    cursor = 0
    for index, token in enumerate(obj["tokens"]):
        token_where = f"{where} tokens[{index}]"
        if not isinstance(token, dict):
            raise DecodeError(f"{token_where}: must be an object")
        token_id = token.get("id")
        start, end = token.get("start"), token.get("end")
        for name, value in (("start", start), ("end", end)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise DecodeError(f"{token_where}: '{name}' must be an integer")
        if not (0 <= start <= end <= byte_length):
            raise DecodeError(f"{token_where}: range [{start}, {end}) outside record")
        if start != cursor:
            raise DecodeError(f"{token_where}: token ranges do not tile the record")
        cursor = end
        ids.append(token_id)
    if cursor != byte_length:
        raise DecodeError(f"{where}: token ranges do not cover the whole record")
    return codec.decode_ids_to_bytes(ids)


def _cmd_decode(args: argparse.Namespace) -> int:
    codec = BPECodec.from_artifacts(args.artifacts)
    decoded: list[bytes] = []
    with open(args.input, "r", encoding="utf-8", errors="strict") as handle:
        for lineno, raw in enumerate(handle, 1):
            line = raw.strip()
            if not line:
                continue
            where = f"{args.input}:{lineno}"
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DecodeError(f"{where}: invalid JSON: {exc}") from exc
            decoded.append(_decode_record(codec, obj, where))
    if args.raw:
        Path(args.raw).write_bytes(b"".join(decoded))
    else:
        out_handle = open(args.out, "w", encoding="utf-8") if args.out else sys.stdout
        try:
            for index, data in enumerate(decoded):
                text = data.decode("utf-8", errors="strict")
                out_handle.write(
                    json.dumps({"record": index, "text": text}, ensure_ascii=False)
                    + "\n"
                )
        finally:
            if args.out:
                out_handle.close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bpe",
        description="Byte-level BPE: incremental training, rank-based encoding, "
        "byte-exact decoding.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="train merges and write artifacts")
    p_train.add_argument("--corpus", required=True, help="JSONL training corpus")
    p_train.add_argument("--config", required=True, help="training config JSON")
    p_train.add_argument("--out", required=True, help="artifact output directory")
    p_train.set_defaults(func=_cmd_train)

    p_check = sub.add_parser("precheck", help="validate an artifact directory")
    p_check.add_argument("--artifacts", required=True)
    p_check.set_defaults(func=_cmd_precheck)

    p_encode = sub.add_parser("encode", help="encode text with existing artifacts")
    p_encode.add_argument("--artifacts", required=True)
    source = p_encode.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="file encoded as a single record")
    source.add_argument("--jsonl", help="JSONL file of {\"text\": ...} records")
    p_encode.add_argument("--out", help="output JSONL path (default: stdout)")
    p_encode.set_defaults(func=_cmd_encode)

    p_decode = sub.add_parser("decode", help="decode an encode-output JSONL file")
    p_decode.add_argument("--artifacts", required=True)
    p_decode.add_argument("--input", required=True, help="encode-output JSONL")
    p_decode.add_argument("--out", help="decoded JSONL path (default: stdout)")
    p_decode.add_argument("--raw", help="write concatenated record bytes to this path")
    p_decode.set_defaults(func=_cmd_decode)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (
        CorpusError,
        ArtifactError,
        DecodeError,
        UnicodeDecodeError,
        UnicodeEncodeError,
        OSError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
