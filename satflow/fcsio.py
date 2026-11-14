"""FCS 3.1 loading with strict structural validation.

Only the narrow input contract of this project is accepted: FCS version
3.1, list mode, little-endian 32-bit float DATA, exactly the four detector
channels named in the plan (matched by ``$PnN``, never by array position).
Anything else raises :class:`~satflow.errors.StructureError` so that an
illegal file can never be silently re-interpreted.
"""

from __future__ import annotations

import numpy as np

from .errors import StructureError

EXPECTED_VERSION = "FCS3.1"
EXPECTED_DATATYPE = "F"          # float32 in the FCS 3.1 spec
EXPECTED_BYTEORD = "1,2,3,4"     # little endian
EXPECTED_MODE = "L"              # list mode
EXPECTED_PAR = 4
EXPECTED_DTYPE = np.float32


def load_fcs(path: str, detectors: list[str]) -> tuple[np.ndarray, list[str], dict]:
    """Load and validate an FCS file.

    Returns ``(data, channel_names, text)`` where ``data`` is an ``(n, 4)``
    float32 matrix ordered exactly like ``detectors`` and ``channel_names``
    are the matched ``$PnN`` names.  Rows keep their original zero-based DATA
    segment indices (row 0 is the first stored event).
    """
    # fcsparser only exposes a keyword parser, so read the fixed header
    # ourselves for the version check instead of guessing afterwards.
    with open(path, "rb") as handle:
        header = handle.read(58)
    if len(header) < 6:
        raise StructureError(f"{path}: file shorter than FCS fixed header")
    version = header[:6].decode("ascii", errors="replace").rstrip()
    if version != EXPECTED_VERSION:
        raise StructureError(
            f"{path}: unsupported FCS version {version!r}; only {EXPECTED_VERSION!r} is accepted"
        )

    try:
        import fcsparser
    except ImportError as exc:  # pragma: no cover - environment guard
        raise StructureError(f"fcsparser is required to read {path}: {exc}") from exc

    try:
        # reformat_meta=True strips the raw $PnN keyword names, which we need
        # for channel-name matching, so keep the unmodified TEXT segment.
        meta, dataframe = fcsparser.parse(path, reformat_meta=False)
    except Exception as exc:
        raise StructureError(f"{path}: fcsparser could not parse file: {exc}") from exc

    text = {key: str(meta[key]) for key in sorted(meta) if isinstance(key, str)}
    _check_keyword(path, text, "$DATATYPE", EXPECTED_DATATYPE)
    _check_keyword(path, text, "$BYTEORD", EXPECTED_BYTEORD)
    _check_keyword(path, text, "$MODE", EXPECTED_MODE)
    _check_keyword(path, text, "$PAR", str(EXPECTED_PAR))

    par = _as_int(text.get("$PAR"), path, "$PAR")
    if par != EXPECTED_PAR:
        raise StructureError(f"{path}: $PAR={par}, this tool only supports {EXPECTED_PAR} channels")

    channel_names: list[str] = []
    for index in range(1, par + 1):
        name = text.get(f"$P{index}N")
        if not isinstance(name, str) or not name:
            raise StructureError(f"{path}: missing or empty $P{index}N channel name")
        channel_names.append(name)

    missing = [name for name in detectors if name not in channel_names]
    if missing:
        raise StructureError(
            f"{path}: plan detectors {missing} not found among FCS channels {channel_names}"
        )
    duplicates = {name for name in channel_names if channel_names.count(name) > 1}
    if duplicates:
        raise StructureError(f"{path}: duplicated $PnN channel names: {sorted(duplicates)}")

    try:
        frame = dataframe[detectors]
    except KeyError as exc:
        raise StructureError(f"{path}: channel column lookup failed: {exc}") from exc
    data = frame.to_numpy(copy=True)

    if data.ndim != 2 or data.shape[1] != EXPECTED_PAR:
        raise StructureError(f"{path}: expected (n, {EXPECTED_PAR}) DATA, got shape {data.shape}")
    if data.dtype != EXPECTED_DTYPE:
        raise StructureError(
            f"{path}: DATA dtype is {data.dtype}; only little-endian float32 is accepted"
        )
    if not np.isfinite(data).all():
        bad = int(np.argwhere(~np.isfinite(data))[0, 0])
        raise StructureError(
            f"{path}: event row {bad} contains NaN/Inf; finite float32 observations are required"
        )

    tot = text.get("$TOT")
    if tot is not None and _as_int(tot, path, "$TOT") != data.shape[0]:
        raise StructureError(
            f"{path}: $TOT={tot} disagrees with {data.shape[0]} rows in DATA segment"
        )

    # One more sanity pass on the per-parameter bit width.
    for index in range(1, par + 1):
        bits = text.get(f"$P{index}B")
        if bits is not None and _as_int(bits, path, f"$P{index}B") != 32:
            raise StructureError(f"{path}: $P{index}B={bits}, only 32-bit parameters are accepted")

    return data, list(detectors), text


def _check_keyword(path: str, text: dict, key: str, expected: str) -> None:
    actual = text.get(key)
    if str(actual) != expected:
        raise StructureError(
            f"{path}: keyword {key}={actual!r}, this tool requires {key}={expected!r}"
        )


def _as_int(value, path: str, key: str) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise StructureError(f"{path}: keyword {key}={value!r} is not an integer") from exc
