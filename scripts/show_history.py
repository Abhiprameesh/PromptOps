from app.storage.database import Database


db = Database()

runs = db.get_all_runs()

print("=" * 90)
print("Evaluation History")
print("=" * 90)

if not runs:
    print("No evaluation runs found.")
else:
    print(
        f"{'Run ID':<8}"
        f"{'Timestamp':<22}"
        f"{'Prompt':<10}"
        f"{'Model':<15}"
        f"{'Accuracy':<10}"
        f"{'Passed':<8}"
        f"{'Failed'}"
    )

    print("-" * 90)

    for run in runs:
        run_id, timestamp, prompt, model, accuracy, passed, failed = run

        print(
    f"{run_id:<8}"
    f"{timestamp:<22}"
    f"{prompt:<10}"
    f"{model:<15}"
    f"{accuracy:<10.2f}"
    f"{passed:<8}"
    f"{failed}"
)



db.close()