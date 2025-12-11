#!/usr/bin/env python3
"""
Script to initialize the database tables
"""

import sys
from pathlib import Path

# Add the parent directory to the path so we can import from app
sys.path.insert(0, str(Path(__file__).parent))

from app.database import engine
from app.models import Base

def init_db():
    """Initialize the database by creating all tables"""
    print("Initializing database tables...")

    try:
        # Create all tables defined in the models
        Base.metadata.create_all(bind=engine)
        print("Database tables created successfully!")

        # Verify tables were created
        from sqlalchemy import inspect
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        print(f"Created tables: {tables}")

    except Exception as e:
        print(f"Error initializing database: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    init_db()