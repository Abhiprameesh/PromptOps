from app.storage.database import Database

db = Database()

print("=" * 50)
print("Database created successfully!")
print("=" * 50)

db.close()