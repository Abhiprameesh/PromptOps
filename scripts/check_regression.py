from app.storage.database import Database


db = Database()

runs = db.get_last_two_runs()

print("=" * 70)
print("Regression Detection")
print("=" * 70)

if len(runs) < 2:
    print("Need at least two evaluation runs.")
else:

    current = runs[0]
    previous = runs[1]

    current_accuracy = current[4]
    previous_accuracy = previous[4]

    difference = current_accuracy - previous_accuracy

    print(f"Current Run : {current[0]}")
    print(f"Previous Run: {previous[0]}")

    print(f"\nCurrent Accuracy : {current_accuracy:.2f}%")
    print(f"Previous Accuracy: {previous_accuracy:.2f}%")

    print()

    if difference > 0:
        print(f"🟢 Accuracy Improved by {difference:.2f}%")

    elif difference < 0:
        print(f"🔴 REGRESSION DETECTED")
        print(f"Accuracy dropped by {abs(difference):.2f}%")

    else:
        print("🟡 No change in accuracy")

db.close()