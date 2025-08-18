import pytest
from datasets import load_dataset
from matplotlib.pyplot import findobj

from sal.evaluation.parser import extract_answer, find_box


def test_find_box_simple():
    pred_str = "Therefore, the factored form is boxed{(a + 5)(b + 2)}."
    inner = find_box(pred_str)
    assert inner.replace(" ", "") == "(a+5)(b+2)".replace(" ", "")


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
    for row in ds:
        solution = row["solution"]
        answer = row["answer"]

        # pred = find_box(solution)
        original = find_box(solution)
        pred = extract_answer(solution, "math")

        if pred != answer:
            print(f"{pred} {answer}   original: {original}")

            answer_boxed = extract_answer(r"\boxed{" + answer + "}", "math")
            print(pred == answer_boxed)
    # for solution in solutions:
    #     try:
    #         pred = extract_answer(solution, "math")
    #     except Exception as e:
    #         pytest.fail(
    #             f"extract_answer failed on example with prompt: {solution[:120]}... Error: {e}"
    #         )
    #     assert answer == pred
