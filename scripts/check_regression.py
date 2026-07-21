from app.reporting.regression import RegressionDetector
from app.storage.database import Database

db = Database()

runs = db.get_last_two_runs()

print("=" * 70)
print("Regression Detection")
print("=" * 70)

if len(runs) < 2:
    print("Need at least two evaluation runs.")
else:

    report = RegressionDetector.compare(
        previous_run=runs[1],
        current_run=runs[0],
    )

    print(f"Current Run : {report['current_run']}")
    print(f"Previous Run: {report['previous_run']}")

    print()
    print(f"Current Accuracy : {report['current_accuracy']:.2f}%")
    print(f"Previous Accuracy: {report['previous_accuracy']:.2f}%")

    print()

    if report["status"] == "improved":
        print(f"🟢 Accuracy Improved by {report['difference']:.2f}%")

    elif report["status"] == "regression":
        print(f"🔴 REGRESSION DETECTED")
        print(f"Accuracy dropped by {abs(report['difference']):.2f}%")

    else:
        print("🟡 No change in accuracy")

db.close()