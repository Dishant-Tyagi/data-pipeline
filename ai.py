import re
from ollama import chat

def extract_code(response_text):
    """Extract Python code from AI response (inside ```python ... ``` or plain)."""
    code_blocks = re.findall(r"```(?:python)?\n(.*?)```", response_text, re.DOTALL)
    if code_blocks:
        return code_blocks[0].strip()
    return response_text.strip()  # fallback


def query_ai(prompt, preview, model, error=None, bad_code=None, unchanged=False):
    """Query AI model, optionally with error feedback or unchanged-result feedback."""
    system_msg = (
        "You are a data filtering assistant. "
        "Return ONLY executable Python code that modifies 'df'. "
        "Do not include explanations, comments, or markdown."
    )

    user_msg = f"Dataset preview:\n{preview}\n\nInstruction: {prompt}\n\n"

    if error and bad_code:
        user_msg += (
            f"The last code you gave caused this error:\n{error}\n\n"
            f"Here was your code:\n{bad_code}\n\n"
            "Please correct it and return only valid Python code that works."
        )

    if unchanged and bad_code:
        user_msg += (
            f"The last code you gave executed without errors, "
            f"but it did not change the dataset (df remained the same).\n\n"
            f"Here was your code:\n{bad_code}\n\n"
            "Please update it so that the dataframe is modified according to the instruction."
        )

    response = chat(
        model=model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ]
    )
    return response["message"]["content"]


def ai_filtering(df, prompt, model="qwen2.5-coder:7b"):
    """
    Filter or transform dataset based on a natural language prompt using AI 
    with self-correction (handles errors and unchanged outputs).
    """
    ai_log = []
    preview = df.head(5).to_string()

    attempts = 0
    max_attempts = 5  # increased because we now also retry on unchanged results
    last_error, last_code = None, None
    unchanged_flag = False

    while attempts < max_attempts:
        attempts += 1
        ai_log.append(f"[INFO] Attempt {attempts}...")

        raw_code = query_ai(
            prompt,
            preview,
            model,
            error=last_error,
            bad_code=last_code,
            unchanged=unchanged_flag
        )
        code = extract_code(raw_code)
        ai_log.append(f"[DEBUG] Extracted Code (Attempt {attempts}):\n{code}")

        local_env = {"df": df.copy()}
        try:
            exec(code, {}, local_env)
            df_transformed = local_env.get("df", df)

            # ✅ Check if AI actually modified df
            if df_transformed.equals(df):
                unchanged_flag = True
                last_code = code
                last_error = None
                ai_log.append("[WARN] AI returned code but dataframe did not change.")
                if attempts < max_attempts:
                    ai_log.append("[INFO] Asking AI to modify the dataset properly...")
                    continue
                else:
                    ai_log.append("[ERROR] Max retries reached (unchanged result). Returning original dataframe.")
                    return df, ai_log

            # ✅ Success case
            ai_log.append("[INFO] Applied AI transformation successfully.")
            ai_log.append(f"[INFO] Final dataset shape after AI: {df_transformed.shape}")
            return df_transformed, ai_log

        except Exception as e:
            last_error, last_code = str(e), code
            unchanged_flag = False  # reset unchanged flag when error occurs
            ai_log.append(f"[ERROR] Execution failed: {e}")
            if attempts < max_attempts:
                ai_log.append("[INFO] Asking AI to self-correct after error...")
            else:
                ai_log.append("[ERROR] Max retries reached (error case). Returning original dataframe.")
                return df, ai_log
