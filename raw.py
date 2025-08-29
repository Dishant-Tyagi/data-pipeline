import subprocess
import time
import os

def generate_raw_csv(folder="D:/assignement"):
    prompt = """
    Generate a raw CSV dataset with messy data.do not add summary just plain csv
    Include: blank values, multiple exact duplicate rows, duplicate column names, extra column.
    Columns: ID, Name, Age, Email, Extra, Email, city, gender, timestamp
    Rows: 15–20
    """

    os.makedirs(folder, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(folder, f"raw_data_{timestamp}.csv")

    print("🚀 Starting Ollama... (first run may take longer)")
    start = time.time()

    # FIX: force UTF-8 decoding
    result = subprocess.run(
        ["ollama", "run", "llama3:8b"],
        input=prompt,
        text=True,
        encoding="utf-8",
        errors="replace",   # replaces bad chars instead of crashing
        capture_output=True
    )

    end = time.time()
    print(f"✅ Ollama finished in {end-start:.2f} seconds")

    csv_text = result.stdout.strip()

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(csv_text)

    print(f"📂 Raw CSV saved at: {output_file}")

# Run function
generate_raw_csv()
