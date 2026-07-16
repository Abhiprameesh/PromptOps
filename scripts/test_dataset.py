from app.evaluation.loader import load_dataset

dataset = load_dataset("datasets/v1/github_issues.json")

print("=" * 60)
print("DATASET LOADED SUCCESSFULLY")
print("=" * 60)

print(f"Version      : {dataset.version}")
print(f"Description  : {dataset.description}")
print(f"Created At   : {dataset.created_at}")
print(f"Total Cases  : {len(dataset.cases)}")

print("\nFirst Test Case")
print("-" * 60)

first_case = dataset.cases[0]

print(f"ID          : {first_case.id}")
print(f"Title       : {first_case.input.title}")
print(f"Body        : {first_case.input.body}")
print(f"Category    : {first_case.expected.category}")
print(f"Priority    : {first_case.expected.priority}")
print(f"Summary     : {first_case.expected.summary}")
print(f"Difficulty  : {first_case.metadata.difficulty}")
print(f"Tags        : {', '.join(first_case.metadata.tags)}")
print(f"Notes       : {first_case.metadata.notes}")