import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

import numpy as np
from sklearn.preprocessing import StandardScaler
import pymongo
from pymongo import MongoClient



#----------------------------loading--------------------------------------
def load_data(path):
    """
    Load data from a CSV, Excel, or JSON file into a pandas DataFrame.
    Args:
        path (str): Path to the file
    Returns:
        pd.DataFrame: Loaded dataset or None if failed
    """
    try:
        ext = os.path.splitext(path)[1].lower()
        
        if ext == ".csv":
            # error_bad_lines=False is deprecated → use on_bad_lines='skip'
            df = pd.read_csv(path, on_bad_lines='skip')
        elif ext in [".xls", ".xlsx"]:
            df = pd.read_excel(path)
        elif ext == ".json":
            df = pd.read_json(path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

        # Log basic info
        print(f"[INFO] Data loaded from {path}")
        print(f"[INFO] Shape: {df.shape} (rows, columns)")
        print(f"[INFO] Columns: {list(df.columns)}")

        return df

    except FileNotFoundError:
        print(f"[ERROR] File not found at {path}")
    except Exception as e:
        print(f"[ERROR] Failed to load data: {e}")

    return None

        
        
        
#--------------------cleaning------------------------------------------

def clean_data_interactive(df):
    print("\n--- Data Cleaning Options ---")

    # 1. Handle missing values
    print("\nHandle missing values:")
    print("1. Drop rows with missing values")
    print("2. Fill missing values with mean/median/mode")
    print("3. Fill with constant value")
    print("4. Do nothing")
    choice = input("Enter choice (1/2/3/4): ").strip()

    if choice == "1":
        df = df.dropna()
        print("[INFO] Dropped rows with missing values")
    elif choice == "2":
        for col in df.columns:
            if df[col].dtype in ["float64", "int64"]:
                df[col] = df[col].fillna(df[col].mean())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])
        print("[INFO] Filled missing values with mean/median/mode")
    elif choice == "3":
        constant = input("Enter constant value to fill missing: ")
        df = df.fillna(constant)
        print(f"[INFO] Filled missing values with constant: {constant}")
    else:
        print("[INFO] Skipped missing value handling")

    # 2. Remove duplicates
    remove_dupes = input("\nRemove duplicate rows? (y/n): ").strip().lower()
    if remove_dupes == "y":
        before = df.shape[0]
        df = df.drop_duplicates()
        after = df.shape[0]
        print(f"[INFO] Removed {before - after} duplicate rows")
    else:
        print("[INFO] Skipped duplicate removal")

    # 3. Handle inconsistent data (optional: case, trim spaces)
    fix_text = input("\nFix text columns (trim spaces, lowercase)? (y/n): ").strip().lower()
    if fix_text == "y":
        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].astype(str).str.strip().str.lower()
        print("[INFO] Cleaned text columns (spaces trimmed, lowercased)")
    else:
        print("[INFO] Skipped text cleaning")

    # 4. Drop constant columns
    drop_const = input("\nDrop constant (same value) columns? (y/n): ").strip().lower()
    if drop_const == "y":
        before = df.shape[1]
        df = df.loc[:, (df != df.iloc[0]).any()]
        after = df.shape[1]
        print(f"[INFO] Dropped {before - after} constant columns")
    else:
        print("[INFO] Skipped constant column removal")

    print(f"[INFO] Final shape after cleaning: {df.shape}")
    return df


#----------------transform----------------------



def transform_data_interactive(df):
    """Ask user whether to transform data, and if yes, apply transformations."""
    choice = input("\nDo you want to transform the dataset? (y/n): ").strip().lower()

    if choice == "n":
        print("[INFO] Skipped all transformations")
        return df  # Return as-is

    print("[INFO] Starting transformations...")

    # --- Scaling ---
    print("\nChoose scaling method for numeric columns:")
    print("1. Min-Max Scaling")
    print("2. Standard Scaling")
    print("3. None")
    scale_choice = input("Enter choice (1/2/3): ").strip()

    if scale_choice == "1":
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        df[df.select_dtypes(include=['int64', 'float64']).columns] = scaler.fit_transform(
            df.select_dtypes(include=['int64', 'float64'])
        )
        print("[INFO] Applied Min-Max Scaling")
    elif scale_choice == "2":
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        df[df.select_dtypes(include=['int64', 'float64']).columns] = scaler.fit_transform(
            df.select_dtypes(include=['int64', 'float64'])
        )
        print("[INFO] Applied Standard Scaling")
    else:
        print("[INFO] Skipped scaling")

    # --- Outlier Handling ---
    print("\nHandle outliers?")
    print("1. Cap values (IQR method)")
    print("2. Remove rows with outliers")
    print("3. None")
    outlier_choice = input("Enter choice (1/2/3): ").strip()

    if outlier_choice == "1":
        for col in df.select_dtypes(include=['int64', 'float64']).columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            df[col] = df[col].clip(lower, upper)
        print("[INFO] Capped outliers using IQR method")
    elif outlier_choice == "2":
        for col in df.select_dtypes(include=['int64', 'float64']).columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            df = df[(df[col] >= lower) & (df[col] <= upper)]
        print("[INFO] Removed rows with outliers")
    else:
        print("[INFO] Skipped outlier handling")

    # --- Encoding ---
    encode_choice = input("\nEncode categorical columns? (y/n): ").strip().lower()
    if encode_choice == "y":
        df = pd.get_dummies(df, drop_first=True)
        print("[INFO] Encoded categorical columns")
    else:
        print("[INFO] Skipped categorical encoding")

    # --- Datetime Expansion ---
    datetime_choice = input("\nExpand datetime columns into year, month, day? (y/n): ").strip().lower()
    if datetime_choice == "y":
        for col in df.select_dtypes(include=['datetime64']).columns:
            df[col+"_year"] = df[col].dt.year
            df[col+"_month"] = df[col].dt.month
            df[col+"_day"] = df[col].dt.day
        print("[INFO] Expanded datetime columns")
    else:
        print("[INFO] Skipped datetime expansion")

    return df



#---------------saving-----------------


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



#-----------------main-----------------


def main():
    # Step 1: Load
    path = input("Enter the file path (CSV, Excel, or JSON): ").strip()
    df = load_data(path)

    if df is None:
        print("[ERROR] Could not load data. Exiting...")
        return

    print("\nPreview of data:")
    print(df.head())

    # Step 2: Clean
    df_clean = clean_data_interactive(df)

    # Step 3: (AI-based enrichment - to be added later)

    # Step 4: Transform (optional)
    choice = input("\nDo you want to transform the data? (y/n): ").strip().lower()
    if choice == "y":
        df_transformed = transform_data_interactive(df_clean)
    else:
        df_transformed = df_clean
        print("[INFO] Skipped transformations")

    # Step 5: Save (File or MongoDB)
    save_data(df_transformed)


if __name__ == "__main__":
    main()
