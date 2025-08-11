from collections import Counter

from sal.config import DatasetConfig
from sal.utils.data import get_dataset

dataset_config = DatasetConfig(
    dataset_name="HuggingFaceH4/MATH-500",
    num_samples=100,
)

dataset = get_dataset(dataset_config)

# Count questions by level
level_counts = Counter(dataset["level"])

print("Questions by level:")
for level, count in sorted(level_counts.items()):
    print(f"Level {level}: {count} questions")

print(f"\nTotal questions: {sum(level_counts.values())}")
