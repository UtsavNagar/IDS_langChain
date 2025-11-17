import os
import pickle
import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download, InferenceClient

# ------------------------------------------
# Load environment variables
# ------------------------------------------
load_dotenv()
hf_token = os.getenv("HF_TOKEN")


# ------------------------------------------
# Load IDS Model (.pkl)
# ------------------------------------------
model_path = hf_hub_download(
    repo_id="utsavNagar/cyberids-ml",
    filename="ids_model.pkl",
    token=hf_token
)

with open(model_path, "rb") as f:
    ids_model = pickle.load(f)


# ------------------------------------------
# Predict from LightGBM IDS
# ------------------------------------------
def predict_intrusion(data: dict) -> str:
    df = pd.DataFrame([data])
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)
    prob = ids_model.predict(df)[0]
    return "Attack" if prob > 0.5 else "Normal"


# ------------------------------------------
# LLM (Chat Completion API)
# ------------------------------------------
LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

client = InferenceClient(model=LLM_MODEL, token=hf_token)


def llm_chat(prompt: str) -> str:
    """Use HF chat completion API (works for Mistral instruct models)."""

    try:
        response = client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
            temperature=0.4
        )

        return response.choices[0].message["content"]

    except Exception as e:
        return f"[LLM ERROR]: {str(e)}"


# ------------------------------------------
# Full analysis pipeline
# ------------------------------------------
def analyze_traffic(features: dict):
    prediction = predict_intrusion(features)

    prompt = f"""
You are a cybersecurity analyst.

Network Data:
{features}

IDS Prediction: {prediction}

Give a detailed analysis including:
1. Whether this is an attack or normal.
2. Why this decision was made.
3. Most likely attack type (DoS / Probe / R2L / U2R).
4. Severity rating.
5. Immediate recommended actions.
6. Final incident summary (2-3 sentences).
"""

    explanation = llm_chat(prompt)
    return prediction, explanation


# ------------------------------------------
# Test Run (example)
# ------------------------------------------
if __name__ == "__main__":
    sample = {
        "duration": 0,
        "protocol_type": 0,
        "service": 52,
        "flag": 9,
        "src_bytes": 181,
        "dst_bytes": 5450,
        "land": 0,
        "wrong_fragment": 0,
        "urgent": 0,
        "hot": 0,
        "num_failed_logins": 0,
        "logged_in": 1,
        "num_compromised": 0,
        "root_shell": 0,
        "su_attempted": 0,
        "num_root": 0,
        "num_file_creations": 0,
        "num_shells": 0,
        "num_access_files": 0,
        "num_outbound_cmds": 0,
        "is_host_login": 0,
        "is_guest_login": 0,
        "count": 10,
        "srv_count": 10,
        "serror_rate": 0.0,
        "srv_serror_rate": 0.0,
        "rerror_rate": 0.0,
        "srv_rerror_rate": 0.0,
        "same_srv_rate": 1.0,
        "diff_srv_rate": 0.0,
        "srv_diff_host_rate": 0.0,
        "dst_host_count": 255,
        "dst_host_srv_count": 255,
        "dst_host_same_srv_rate": 1.0,
        "dst_host_diff_srv_rate": 0.0,
        "dst_host_same_src_port_rate": 0.0,
        "dst_host_srv_diff_host_rate": 0.0,
        "dst_host_serror_rate": 0.0,
        "dst_host_srv_serror_rate": 0.0,
        "dst_host_rerror_rate": 0.0,
        "dst_host_srv_rerror_rate": 0.0
    }

    pred, report = analyze_traffic(sample)
    print("\nPrediction:", pred)
    print("\n=== Incident Report ===\n")
    print(report)
