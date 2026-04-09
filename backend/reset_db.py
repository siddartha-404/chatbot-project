from database.database import engine
from database import models

print("Connecting to database and dropping old tables...")
models.Base.metadata.drop_all(bind=engine)

print("Creating brand new, updated tables...")
models.Base.metadata.create_all(bind=engine)

print("✅ Success! Database is completely reset and ready")