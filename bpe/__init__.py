"""Byte-level BPE: incremental training, rank-based encoding, exact decoding.

Pure Python 3.11 standard library.  No finished tokenizer library is used.
"""

from .core import BASE_VOCAB_SIZE, SymbolTable, segment_byte_spans, whitespace_run_spans
from .trainer import IncrementalBPETrainer, TrainParams
from .encoder import BPECodec, DecodeError, Token
from .artifacts import ArtifactError, load_model, precheck, save_artifacts
from .corpus import CorpusError, read_config, read_corpus

__version__ = "1.0.0"

__all__ = [
    "BASE_VOCAB_SIZE",
    "SymbolTable",
    "segment_byte_spans",
    "whitespace_run_spans",
    "IncrementalBPETrainer",
    "TrainParams",
    "BPECodec",
    "DecodeError",
    "Token",
    "ArtifactError",
    "load_model",
    "precheck",
    "save_artifacts",
    "CorpusError",
    "read_config",
    "read_corpus",
    "__version__",
]
