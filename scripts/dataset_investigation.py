from collections import Counter

from sal.config import DatasetConfig
from sal.utils.data import get_dataset
from sal.utils.experiment import get_math500_indices

# dataset_config = DatasetConfig(
#     dataset_name="HuggingFaceH4/MATH-500",
#     num_samples=100,
# )
dataset_config = DatasetConfig(
    benchmark_name="HuggingFaceH4/MATH-500",
    benchmark_indicies=get_math500_indices(subset="hard"),
)

dataset = get_dataset(dataset_config)

# Count questions by level
level_counts = Counter(dataset["level"])

print("Questions by level:")
for level, count in sorted(level_counts.items()):
    print(f"Level {level}: {count} questions")

print(f"\nTotal questions: {sum(level_counts.values())}")

# All questions
# Questions by level:
# Level 1: 43 questions
# Level 2: 90 questions
# Level 3: 105 questions
# Level 4: 128 questions
# Level 5: 134 questions
# Total questions: 500
# ---
# First 100 questions
# Questions by level:
# Level 1: 11 questions
# Level 2: 25 questions
# Level 3: 19 questions
# Level 4: 22 questions
# Level 5: 23 questions
# Total questions: 100
# ---
# Hard questions:
# Questions by level:
# Level 1: 3 questions
# Level 2: 5 questions
# Level 3: 7 questions
# Level 4: 22 questions
# Level 5: 49 questions
# Total questions: 86
