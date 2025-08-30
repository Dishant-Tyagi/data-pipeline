import pandas as pd

def interactive_clean_data(df):
    """Interactive data cleaning with user choices before AI filtering."""
    cleaning_log = []  # collect actions for logging

    print("\n--- Data Cleaning Menu ---")
    print("Choose what cleaning steps you want to perform:")
    print("1. Fill missing values")
    print("2. Remove duplicate rows")
    print("3. Standardize text columns (lowercase + trim spaces)")
    print("4. Drop constant (same value) columns")
    print("5. Skip cleaning")
    print("6. Run all steps")

    choices = input("Enter numbers separated by commas (e.g., 1,3,4): ").split(",")
    choices = [c.strip() for c in choices]

    try:
        # 1. Fill missing values
        if "1" in choices or "6" in choices:
            for col in df.columns:
                try:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        df[col] = df[col].fillna(df[col].mean())
                    else:
                        df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "")
                except Exception as e:
                    cleaning_log.append(f"[WARN] Could not fill column {col}: {e}")
            cleaning_log.append("[INFO] Missing values handled")

        # 2. Remove duplicates
        if "2" in choices or "6" in choices:
            before = df.shape[0]
            df = df.drop_duplicates()
            removed = before - df.shape[0]
            cleaning_log.append(f"[INFO] Removed {removed} duplicate rows")

        # 3. Standardize text
        if "3" in choices or "6" in choices:
            for col in df.select_dtypes(include=["object"]).columns:
                try:
                    df[col] = df[col].astype(str).str.strip().str.lower()
                except Exception as e:
                    cleaning_log.append(f"[WARN] Could not clean text in {col}: {e}")
            cleaning_log.append("[INFO] Text columns standardized")

        # 4. Drop constant columns
        if "4" in choices or "6" in choices:
            before = df.shape[1]
            if not df.empty:
                df = df.loc[:, df.apply(pd.Series.nunique) > 1]
            dropped = before - df.shape[1]
            cleaning_log.append(f"[INFO] Dropped {dropped} constant columns")

        # 5. Skip cleaning
        if "5" in choices and "6" not in choices:
            cleaning_log.append("[INFO] Skipped cleaning")

    except Exception as e:
        cleaning_log.append(f"[ERROR] Cleaning failed: {e}")

    cleaning_log.append(f"[INFO] Final shape after cleaning: {df.shape}")

    # Print log summary at the end
    print("\n--- Cleaning Log ---")
    for log in cleaning_log:
        print(log)

    return df, cleaning_log


def save_log(log, filename="cleaning_log.txt"):
    """Save cleaning log into a text file."""
    with open(filename, "w", encoding="utf-8") as f:
        for entry in log:
            f.write(entry + "\n")
    print(f"[INFO] Cleaning log saved to {filename}")
