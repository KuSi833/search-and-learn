from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Set
import sys


@dataclass
class FusionRerun:
    base_run_id: str
    rerun_run_id: str
    coverage: int


# List of FusionRerun objects based on your data
fusion_reruns = [
    FusionRerun(base_run_id="0hermenf", rerun_run_id="77pyab58", coverage=20),
    FusionRerun(base_run_id="58xqqffr", rerun_run_id="gfw8x07r", coverage=20),
    FusionRerun(base_run_id="6nuelycf", rerun_run_id="tqfyvf5w", coverage=20),
    FusionRerun(base_run_id="defwqu3v", rerun_run_id="77pyab58", coverage=20),
    FusionRerun(base_run_id="dtjaqwyg", rerun_run_id="tqfyvf5w", coverage=20),
    FusionRerun(base_run_id="whldvunb", rerun_run_id="tqfyvf5w", coverage=20),
    FusionRerun(base_run_id="22z91qa7", rerun_run_id="tqfyvf5w", coverage=10),
    FusionRerun(base_run_id="2jcdgx7c", rerun_run_id="tqfyvf5w", coverage=10),
    FusionRerun(base_run_id="8ff83v7m", rerun_run_id="77pyab58", coverage=10),
    FusionRerun(base_run_id="8yyge5wj", rerun_run_id="gfw8x07r", coverage=10),
    FusionRerun(base_run_id="ch6s6c0c", rerun_run_id="tqfyvf5w", coverage=10),
    FusionRerun(base_run_id="ylsdosex", rerun_run_id="77pyab58", coverage=10),
]


# --- Helper utilities ---

OUTPUT_DIR = Path("./output")
EXPECTED_FILENAME = "inference_output.jsonl"


def _expected_output_path(run_id: str) -> Path:
    return OUTPUT_DIR / run_id / EXPECTED_FILENAME


def _unique_run_ids(pairs: Iterable[FusionRerun]) -> Set[str]:
    ids: Set[str] = set()
    for pr in pairs:
        ids.add(pr.base_run_id)
        ids.add(pr.rerun_run_id)
    return ids


def check_all_runs_exist(pairs: Iterable[FusionRerun]) -> None:
    """Verify required output files exist for all run IDs.

    Prints any missing run IDs with the expected file path. Exits with code 1 if any
    files are missing. Does nothing (and prints a short confirmation) if all exist.
    """
    unique_ids = _unique_run_ids(pairs)

    missing: Dict[str, Path] = {}
    for run_id in unique_ids:
        path = _expected_output_path(run_id)
        if not path.exists():
            missing[run_id] = path

    if missing:
        print("Missing run outputs detected:")
        for rid, p in sorted(missing.items()):
            print(f"- {rid}: expected file not found at {p}")
        sys.exit(1)

    print(f"All runs present ({len(unique_ids)} unique run IDs).")


if __name__ == "__main__":
    check_all_runs_exist(fusion_reruns)
