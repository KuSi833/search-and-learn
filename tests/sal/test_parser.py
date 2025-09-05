import pytest
from datasets import load_dataset

from sal.evaluation.parser import extract_answer, find_box


def test_find_box_simple():
    pred_str = "Therefore, the factored form is boxed{(a + 5)(b + 2)}."
    inner = find_box(pred_str)
    assert inner.replace(" ", "") == "(a+5)(b+2)".replace(" ", "")


def test_extract_answer_fraction():
    pred_str = r"The answer is \boxed{\frac{243}{625}}."
    answer_str = r"\frac{243}{625}"
    extracted = extract_answer(pred_str, "math")
    assert extracted == r"\frac{243}{625}"


@pytest.mark.slow
@pytest.mark.parametrize("row", load_dataset("HuggingFaceH4/MATH-500", split="test"))
def test_extract_answer_math500_entire_dataset(row):
    """Ensure extract_answer does not raise across the full MATH-500 test split."""
    solution = row["solution"]
    answer = row["answer"]

    pred = find_box(solution)

    assert pred == answer


# def _custom_strip(s: str) -> str:


if __name__ == "__main__":
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    for idx, row in enumerate(ds):
        print(
            f"{idx}: {row['unique_id']}",
        )
