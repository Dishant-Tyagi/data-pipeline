import pandas as pd

def load_data():
    """Load data from CSV, Excel, or JSON with logs (printed only)."""
    load_log = []

    filepath = input("Enter file path to load: ")

    try:
        if filepath.endswith(".csv"):
            df = pd.read_csv(filepath)
            load_log.append(f"[INFO] Loaded CSV file: {filepath}")

        elif filepath.endswith(".xlsx") or filepath.endswith(".xls"):
            df = pd.read_excel(filepath)
            load_log.append(f"[INFO] Loaded Excel file: {filepath}")

        elif filepath.endswith(".json"):
            df = pd.read_json(filepath)
            load_log.append(f"[INFO] Loaded JSON file: {filepath}")

        else:
            load_log.append(f"[ERROR] Unsupported file format: {filepath}")
            return None, load_log

        load_log.append(f"[INFO] Dataset shape: {df.shape}")
        return df, load_log

    except Exception as e:
        load_log.append(f"[ERROR] Failed to load {filepath}: {e}")
        return None, load_log
