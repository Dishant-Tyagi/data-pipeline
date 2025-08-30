import pandas as pd

def interactive_transform_data(df):
    """Interactively apply transformations to the DataFrame."""
    transform_log = []

    print("\n--- Transformation Options ---")
    print("1. Rename columns")
    print("2. Convert column to datetime")
    print("3. One-hot encode categorical columns")
    print("4. Scale numeric columns (min-max)")
    print("5. Skip transformations")

    while True:
        choice = input("\nChoose an option (1-5): ").strip()

        if choice == "1":
            print(f"Current columns: {list(df.columns)}")
            old_name = input("Enter column name to rename: ").strip()
            new_name = input("Enter new column name: ").strip()
            if old_name in df.columns:
                df.rename(columns={old_name: new_name}, inplace=True)
                transform_log.append(f"[INFO] Renamed column '{old_name}' → '{new_name}'")
            else:
                transform_log.append(f"[WARN] Column '{old_name}' not found")

        elif choice == "2":
            col = input("Enter column name to convert to datetime: ").strip()
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
                transform_log.append(f"[INFO] Converted '{col}' to datetime")
            else:
                transform_log.append(f"[WARN] Column '{col}' not found")

        elif choice == "3":
            col = input("Enter categorical column to one-hot encode: ").strip()
            if col in df.columns:
                df = pd.get_dummies(df, columns=[col])
                transform_log.append(f"[INFO] One-hot encoded column '{col}'")
            else:
                transform_log.append(f"[WARN] Column '{col}' not found")

        elif choice == "4":
            col = input("Enter numeric column to scale (min-max): ").strip()
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
                transform_log.append(f"[INFO] Scaled column '{col}' using min-max")
            else:
                transform_log.append(f"[WARN] Column '{col}' not numeric or not found")

        elif choice == "5":
            transform_log.append("[INFO] Skipped transformations")
            break

        else:
            print("Invalid choice, try again.")

        cont = input("Apply another transformation? (y/n): ").strip().lower()
        if cont != "y":
            break

    transform_log.append(f"[INFO] Final dataset shape after transformations: {df.shape}")
    return df, transform_log
