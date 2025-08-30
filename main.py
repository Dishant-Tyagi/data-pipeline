from loader import load_data
from cleaner import interactive_clean_data
from transformer import interactive_transform_data
from ai import ai_filtering
from saver import save_data

def main():
    df, _ = load_data()
    print(f"[INFO] Data loaded. Shape: {df.shape}")

    df, _ = interactive_clean_data(df)
    print(f"[INFO] Cleaning done. Shape: {df.shape}")

    df, _ = interactive_transform_data(df)
    print(f"[INFO] Transformation done. Shape: {df.shape}")

    prompt = input("Enter AI transformation prompt: ")
    df, _ = ai_filtering(df, prompt)
    print(f"[INFO] AI transformation done. Shape: {df.shape}")

    save_data(df)   # ✅ only pass df
    print(f"[INFO] Data saved successfully.")

if __name__ == "__main__":
    main()
