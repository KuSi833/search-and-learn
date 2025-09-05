#!/usr/bin/env python3
"""
Test script to demonstrate canonical form debugging functionality.
Run this to see how the canonical form grouping works with debugging enabled.
"""

from src.sal.utils.math import test_canonical_grouping


def main():
    print("Testing canonical form grouping with debugging...")

    # Example 1: Basic mathematical equivalence
    print("\n" + "=" * 60)
    print("EXAMPLE 1: Basic Mathematical Equivalence")
    print("=" * 60)

    answers1 = ["5", "2+3", "5.0", "10/2", "6"]
    scores1 = [0.9, 0.8, 0.7, 0.6, 0.85]
    test_canonical_grouping(answers1, scores1)

    # Example 2: Fractions and decimals
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Fractions and Decimals")
    print("=" * 60)

    answers2 = ["1/2", "0.5", "2/4", "0.50", "3/4"]
    scores2 = [0.9, 0.8, 0.7, 0.6, 0.85]
    test_canonical_grouping(answers2, scores2)

    # Example 3: Algebraic expressions
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Algebraic Expressions")
    print("=" * 60)

    answers3 = ["x^2", "x*x", "(x)^2", "x**2", "y^2"]
    scores3 = [0.9, 0.8, 0.7, 0.6, 0.85]
    test_canonical_grouping(answers3, scores3)

    # Example 4: When canonical form fails
    print("\n" + "=" * 60)
    print("EXAMPLE 4: When Canonical Form Fails (complex expressions)")
    print("=" * 60)

    answers4 = ["\\int x dx", "x^2/2 + C", "complex_expression", "another_complex"]
    scores4 = [0.9, 0.8, 0.7, 0.6]
    test_canonical_grouping(answers4, scores4)


if __name__ == "__main__":
    main()
