from collections import Counter

from sal.config import DatasetConfig
from sal.utils.data import get_dataset
from sal.utils.experiment import get_math500_indices

# dataset_config = DatasetConfig(
#     dataset_name="HuggingFaceH4/MATH-500",
#     num_samples=100,
# )
dataset_config = DatasetConfig(
    dataset_name="HuggingFaceH4/MATH-500",
    dataset_indicies=get_math500_indices(subset="hard"),
)

dataset = get_dataset(dataset_config)

# Count questions by level
level_counts = Counter(dataset["level"])

print("Questions by level:")
for level, count in sorted(level_counts.items()):
    print(f"Level {level}: {count} questions")

print(f"\nTotal questions: {sum(level_counts.values())}")

# Questions by level:
# Level 1: 43 questions
# Level 2: 90 questions
# Level 3: 105 questions
# Level 4: 128 questions
# Level 5: 134 questions
# Total questions: 500
# ---
# Questions by level:
# Level 1: 11 questions
# Level 2: 25 questions
# Level 3: 19 questions
# Level 4: 22 questions
# Level 5: 23 questions
# Total questions: 100
