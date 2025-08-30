import pandas as pd
import os
import pymongo
from pymongo import MongoClient

def get_unique_filename(filepath: str) -> str:
    """Generate a unique file name if one already exists."""
    base, ext = os.path.splitext(filepath)
    counter = 1
    new_filepath = filepath

    while os.path.exists(new_filepath):
        new_filepath = f"{base}_{counter}{ext}"
        counter += 1

    return new_filepath


def save_data(df):
    print("\nChoose save option:")
    print("1. Save as CSV")
    print("2. Save as Excel")
    print("3. Save as JSON")
    print("4. Save to MongoDB")
    print("5. Cancel")

    choice = input("Enter choice (1/2/3/4/5): ").strip()

    if choice == "1":
        output_file = input("Enter output file path (with .csv extension): ").strip()
        output_file = get_unique_filename(output_file)
        try:
            df.to_csv(output_file, index=False)
            print(f"[INFO] Saved CSV at {output_file}")
        except Exception as e:
            print(f"[ERROR] Failed to save CSV: {e}")

    elif choice == "2":
        output_file = input("Enter output file path (with .xlsx extension): ").strip()
        output_file = get_unique_filename(output_file)
        try:
            from openpyxl import Workbook
            df.to_excel(output_file, index=False)
            print(f"[INFO] Saved Excel at {output_file}")
        except ImportError:
            print("[ERROR] openpyxl not installed. Run: pip install openpyxl")
        except Exception as e:
            print(f"[ERROR] Failed to save Excel: {e}")

    elif choice == "3":
        output_file = input("Enter output file path (with .json extension): ").strip()
        output_file = get_unique_filename(output_file)
        try:
            df.to_json(output_file, orient="records", indent=4)
            print(f"[INFO] Saved JSON at {output_file}")
        except Exception as e:
            print(f"[ERROR] Failed to save JSON: {e}")

    elif choice == "4":
        try:
            client = MongoClient("mongodb://localhost:27017/")
            db_name = input("Enter MongoDB database name: ").strip() or "my_database"
            collection_name = input("Enter MongoDB collection name: ").strip() or "my_collection"
            db = client[db_name]
            collection = db[collection_name]
            data_dict = df.to_dict("records")
            collection.insert_many(data_dict)
            print(f"[INFO] Data saved to MongoDB ({db_name}.{collection_name})")
        except Exception as e:
            print(f"[ERROR] Failed to save to MongoDB: {e}")

    elif choice == "5":
        print("[INFO] Save cancelled.")
    else:
        print("[ERROR] Invalid choice. Please try again.")
