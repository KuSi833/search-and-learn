#!/usr/bin/env python3
import logging
from dataclasses import asdict
from typing import Any, Dict, Final, List, Tuple

import click
import pyrootutils
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from vllm import LLM  # type: ignore

from sal.config import (
    BaseConfig,
    DatasetConfig,
    ExperimentConfig,
    GeneratorConfig,
    PRMConfig,
    SearchConfig,
)
from sal.models.reward_models import load_prm
from sal.search import best_of_n
from sal.utils.data import get_dataset
from sal.utils.experiment import get_model_base_path
from sal.utils.logging import setup_logging
from sal.utils.qwen_math_parser import extract_answer, math_equal
from sal.utils.score import aggregate_scores

setup_logging()

root = pyrootutils.find_root(indicator="pyproject.toml")
logger = logging.getLogger(__name__)

console = Console()

# --- Typed, versioned configuration ---
MODEL_BASE_PATH: Final = get_model_base_path()

INSTRUCT_MODEL: Final[GeneratorConfig] = GeneratorConfig(
    base_path=MODEL_BASE_PATH,
    name="Qwen/Qwen2.5-Math-7B-Instruct",
    parameter_count="7B",
)

PRM_MODEL: Final[PRMConfig] = PRMConfig(
    base_path=MODEL_BASE_PATH,
    name="Qwen/Qwen2.5-Math-PRM-7B",
)

BASE_CONFIG: Final[BaseConfig] = BaseConfig(
    generator_config=INSTRUCT_MODEL,
    prm_config=PRM_MODEL,
)

# Define strategies to evaluate (edit to sweep hyperparameters)
EXPERIMENTS: Final[List[ExperimentConfig]] = [
    ExperimentConfig(
        approach="best_of_n",
        search_config=SearchConfig(
            n=16,
            temperature=0.7,
            top_p=0.8,
            max_tokens=2048,
            agg_strategy="prod",
            search_batch_size=25,
        ),
    ),
    ExperimentConfig(
        approach="beam_search",
        search_config=SearchConfig(
            n=64,
            temperature=0.7,
            top_p=0.8,
            max_tokens=2048,
            agg_strategy="prod",
            search_batch_size=1,  # beam_search expects 1
        ),
    ),
]


def _evaluate_correct(
    example: Dict[str, Any], pred_text: str, benchmark: str
) -> Tuple[bool, str, str]:
    # Extract gold answer and predicted answer for comparison
    if benchmark == "math":
        gt = str(example.get("answer", ""))
    else:
        # Fallback for other datasets: try common fields
        gt = str(example.get("answer", example.get("gt", "")))
    pred_ans = extract_answer(pred_text, benchmark)
    try:
        ok = bool(math_equal(pred_ans, gt))
    except Exception:
        ok = False
    return ok, pred_ans, gt


@click.command(help="Run full end-to-end strategies on a single dataset index (no W&B)")
@click.option("--index", required=True, type=int, help="Dataset index to evaluate")
def main(index: int) -> None:
    # Configure dataset to a single index
    ds_cfg = DatasetConfig(**asdict(BASE_CONFIG.dataset_config))
    ds_cfg.dataset_indicies = {index}
    dataset = get_dataset(ds_cfg)
    if len(dataset) != 1:
        raise RuntimeError(f"Expected 1 sample after selection; got {len(dataset)}")
    example = dataset[0]

    # Load models once
    llm = LLM(
        model=BASE_CONFIG.generator_config.get_model_path(),
        gpu_memory_utilization=BASE_CONFIG.generator_config.gpu_memory_utilization,
        enable_prefix_caching=True,
        seed=BASE_CONFIG.seed,
        tensor_parallel_size=1,
        max_model_len=BASE_CONFIG.generator_config.max_model_len,
        enforce_eager=True,
    )
    prm = load_prm(BASE_CONFIG.prm_config)

    # Prepare minimal input dict expected by strategy functions
    x = {"problem": [example["problem"]]}

    results: List[Dict[str, Any]] = []

    # Print concise summary
    print(f"Index: {index}")
    print(f"Ground Truth: {example.get('answer')}")

    for exp in EXPERIMENTS:
        logger.info(f"Running approach={exp.approach}")
        cfg = exp.search_config
        console.print(
            Rule(
                title=(
                    f"Approach: {exp.approach} — n={cfg.n}, temp={cfg.temperature}, "
                    f"top_p={cfg.top_p}, max_tokens={cfg.max_tokens}, agg={cfg.agg_strategy}"
                )
            )
        )
        if exp.approach == "best_of_n":
            out = best_of_n(x.copy(), exp, llm, prm)
        # elif exp.approach == "beam_search":
        #     out = beam_search(x.copy(), exp, llm, prm)
        else:
            logger.warning(f"Skipping unsupported approach: {exp.approach}")
            continue

        # Build candidates table: PRM aggregate, tokens, extracted answer, preview
        completions = out.get("completions", [[""]])[0]
        scores = out.get("scores", [[[]]])[0]
        token_counts = out.get("completion_tokens", [[0]])[0]
        agg_scores = (
            [aggregate_scores(s, cfg.agg_strategy) for s in scores] if scores else []
        )
        best_idx = (
            int(max(range(len(agg_scores)), key=lambda i: agg_scores[i]))
            if agg_scores
            else -1
        )

        table = Table(title="Candidates")
        table.add_column("#", justify="right", style="bold")
        table.add_column("PRM", justify="right")
        table.add_column("Tokens", justify="right")
        table.add_column("Answer")
        table.add_column("Preview")

        def _preview(text: str, limit: int = 80) -> str:
            t = text.replace("\n", " ")
            return t[:limit] + ("…" if len(t) > limit else "")

        benchmark = BASE_CONFIG.evaluation_config.benchmark
        for i, comp in enumerate(completions):
            prm_val = f"{agg_scores[i]:.4f}" if agg_scores else "-"
            tok_str = f"{token_counts[i]}" if i < len(token_counts) else "-"
            ans = extract_answer(comp, benchmark)
            row_style = "bold green" if i == best_idx else None
            table.add_row(
                str(i), prm_val, tok_str, ans, _preview(comp), style=row_style
            )

        console.print(table)

        pred_text = out.get("pred", [""])[0]
        ok, pred_ans, gt = _evaluate_correct(
            example, pred_text, BASE_CONFIG.evaluation_config.benchmark
        )

        print(f"Predicted: {pred_ans}")
        console.print(Text(f"Correct: {ok}", style="green" if ok else "red"))
        results.append(
            {
                "approach": exp.approach,
                "search_config": asdict(exp.search_config),
                "correct": ok,
                "pred_extracted": pred_ans,
                "gt": gt,
            }
        )


if __name__ == "__main__":
    main()
