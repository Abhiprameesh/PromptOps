from app.storage.database import Database

db = Database()

summary = db.get_run_summary(3)
details = db.get_run_details(3)

print("=" * 60)
print("Run Summary")
print("=" * 60)
print(summary)

print("\n")

print("=" * 60)
print("Case Details")
print("=" * 60)

for row in details:
    print(row)

db.close()