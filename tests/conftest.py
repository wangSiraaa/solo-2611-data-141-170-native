import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "fixtures"
CASES = ROOT / "cases"
MANIFEST = FIXTURES / "manifest.json"
ADD_FRAGMENT = CASES / "add_fragment.json"
CONFLICT_PATH = "../cases/c1-conflict.csv"


@pytest.fixture(scope="session")
def manifest_path() -> Path:
    return MANIFEST


def write_manifest(dir_path: Path, manifest: dict) -> Path:
    """把 manifest 写入 dir_path（其相对路径相对该目录解析）。"""
    dir_path.mkdir(parents=True, exist_ok=True)
    path = dir_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def make_entry(chain_id, path, start, end, warmup_end, thin=1):
    return {
        "chain_id": chain_id,
        "path": path,
        "start_iteration": start,
        "end_iteration": end,
        "warmup_end": warmup_end,
        "thin": thin,
    }


def write_chain_csv(dir_path: Path, rel_path: str, iterations, values: dict):
    """写一个片段 CSV；values: {param: 与 iterations 等长的序列}。"""
    full = dir_path / rel_path
    full.parent.mkdir(parents=True, exist_ok=True)
    params = list(values)
    lines = ["iteration," + ",".join(params)]
    for i, it in enumerate(iterations):
        lines.append(str(it) + "," + ",".join(str(values[p][i]) for p in params))
    full.write_text("\n".join(lines) + "\n", encoding="utf-8")
