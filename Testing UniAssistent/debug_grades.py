import json
import os
import sys

# Add parent directory to path to load .env
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from uni_db_manager import UniDatabaseManager

def test():
    print("Initializing DB Manager...")
    db = UniDatabaseManager(headless=True)
    print("\nFetching Grades...")
    try:
        grades = db.get_academic_history(force=True)
        print("\nSUCCESS! Here is the parsed AcademicHistoryData:")
        print(grades.model_dump_json(indent=2))
    except Exception as e:
        print(f"\nERROR: {e}")

if __name__ == "__main__":
    test()
