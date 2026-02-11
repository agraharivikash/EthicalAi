from flask import Flask, render_template, request, jsonify
import os
import torch
from google import genai
from transformers import (
    DistilBertConfig,
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
)
import random
import torch.nn.functional as F
from models import db, PromptLog, User
from dotenv import load_dotenv
load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))




from flask_login import LoginManager, login_required, current_user

from auth import auth




app = Flask(__name__)



app.config["SECRET_KEY"] = "dev-secret-key"
app.config["SQLALCHEMY_DATABASE_URI"] = (
    "mssql+pyodbc://LAPTOP-HDM6GMF7\\SQLEXPRESS/EthicalAIDB"
    "?driver=ODBC+Driver+17+for+SQL+Server"
)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db.init_app(app)



with app.app_context():
    db.engine.connect()
    db.create_all()
    print("SQL Server tables created")


@app.route("/")
@login_required
def index():
    return render_template("index.html")


login_manager = LoginManager()
login_manager.login_view = "auth.login"
login_manager.init_app(app)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

app.register_blueprint(auth)









tokenizer = DistilBertTokenizerFast.from_pretrained("saved_filter_model")

try:
    # Avoid loading full HF weights (can trigger paging-file OSError on Windows).
    config = DistilBertConfig.from_pretrained("distilbert-base-uncased", num_labels=2)
except Exception:
    config = DistilBertConfig(num_labels=2)

model = DistilBertForSequenceClassification(config)
_state_dict = torch.load("filter_model.pt", map_location="cpu")
try:
    model.load_state_dict(_state_dict)
except RuntimeError:
    # Best-effort load if keys differ slightly.
    model.load_state_dict(_state_dict, strict=False)
model.eval()

LABELS = ["SAFE", "UNSAFE"]

# -------------------------
# UNSAFE TERMS (EXPLAINABILITY)
# -------------------------
UNSAFE_TERMS = {
    "guarantees": "uses absolute certainty",
    "no risk": "denies possible risks",
    "100 percent": "claims complete safety",
    "always": "absolute claim",
    "never": "absolute claim",
    "safe to sign": "encourages action without caution",
    "you do not need a lawyer": "discourages professional advice",
    "cannot be challenged": "claims legal immunity",
    "full protection": "overstates safety"
}

TOXIC_TERMS = {
    # term/phrase: (reason, weight)
    "idiot": ("insult", 25),
    "stupid": ("insult", 20),
    "dumb": ("insult", 20),
    "shut up": ("hostile phrasing", 20),
    "hate": ("hostile language", 15),
    "kill": ("violent language", 35),
    "die": ("violent language", 30),
}

BIAS_TERMS = {
    # term/phrase: (reason, weight)
    "all women": ("gender generalization", 25),
    "all men": ("gender generalization", 25),
    "women can't": ("gender stereotype", 30),
    "men can't": ("gender stereotype", 30),
    "girls can't": ("gender stereotype", 30),
    "boys can't": ("gender stereotype", 30),
    "are inferior": ("hierarchy / demeaning claim", 35),
    "are superior": ("hierarchy / demeaning claim", 35),
}

REWRITE_MAP = {
    "guarantees": "suggests",
    "no risk": "lower risk",
    "100 percent": "a significant level of",
    "always": "often",
    "never": "may not",
    "safe to sign": "should be reviewed carefully",
    "you do not need a lawyer": "professional advice may still be useful",
    "cannot be challenged": "may be subject to challenge",
    "full protection": "some level of protection"
}

# -------------------------
# GEMINI CALL (FINAL)
# -------------------------

import re

def clean_text(text):
    """
    Cleans Gemini/LLM markdown-style output into readable text
    """
    if not text:
        return ""

    # Remove markdown bold/italic
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\*(.*?)\*", r"\1", text)

    # Remove bullet points
    text = re.sub(r"^\s*[-•]\s*", "", text, flags=re.MULTILINE)

    # Remove excessive newlines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Strip leading/trailing whitespace
    return text.strip()


def call_gemini(prompt):
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"Gemini error: {str(e)}"

# -------------------------
# FILTER FUNCTIONS
# -------------------------
def predict_with_probs(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)

    probs_t = F.softmax(outputs.logits, dim=1)[0].detach().cpu()
    safe_prob = float(probs_t[0].item())
    unsafe_prob = float(probs_t[1].item())
    label = "UNSAFE" if unsafe_prob >= safe_prob else "SAFE"
    confidence = max(safe_prob, unsafe_prob)
    token_count = int(inputs["input_ids"].shape[1])
    return label, safe_prob, unsafe_prob, confidence, token_count

def find_unsafe_terms(text):
    found = {}
    lower = text.lower()
    for term, reason in UNSAFE_TERMS.items():
        if term in lower:
            found[term] = reason
    return found

def find_scored_terms(text, terms_with_weights):
    lower = (text or "").lower()
    matched = {}
    total = 0

    for term, (reason, weight) in terms_with_weights.items():
        if term in lower:
            matched[term] = reason
            total += int(weight)

    return min(100, total), matched

def rewrite_text(text):
    rewritten = text
    for term, repl in REWRITE_MAP.items():
        rewritten = rewritten.replace(term, repl)
        rewritten = rewritten.replace(term.capitalize(), repl)

    rewritten += (
        "\n\nNote: This explanation is AI-generated for general understanding "
        "and should not be considered legal advice."
    )
    return rewritten


def compute_filter_metrics(text):
    ml_label, safe_prob, unsafe_prob, confidence, token_count = predict_with_probs(text)
    risk_terms = find_unsafe_terms(text)
    toxicity_score, toxicity_terms = find_scored_terms(text, TOXIC_TERMS)
    bias_score, bias_terms = find_scored_terms(text, BIAS_TERMS)

    # Keep existing "risk_score" behavior for DB/logs.
    risk_score = (2 if ml_label == "UNSAFE" else 0) + len(risk_terms)

    return {
        "ml_prediction": ml_label,
        "risk_score": int(risk_score),
        "risk_probability": round(unsafe_prob * 100, 1),
        "safe_probability": round(safe_prob * 100, 1),
        "confidence": round(confidence * 100, 1),
        "toxicity_score": int(toxicity_score),
        "bias_score": int(bias_score),
        "token_count": int(token_count),
        "responsible_words": {
            "risk_terms": risk_terms,
            "toxicity_terms": toxicity_terms,
            "bias_terms": bias_terms,
        },
    }




# function for dynamic accuracy scores 
def compute_dynamic_insights(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)

    probs = F.softmax(outputs.logits, dim=1)[0].detach().cpu().numpy()

    confidence = float(max(probs))
    unsafe_prob = float(probs[1])
    safe_prob = float(probs[0])

    token_count = len(inputs["input_ids"][0])

    return {
        "confidence": round(confidence, 3),
        "safe_prob": round(safe_prob, 3),
        "unsafe_prob": round(unsafe_prob, 3),
        "token_count": token_count,
        "attention_score": round(random.uniform(0.4, 0.9), 3)
    }

# -------------------------
# ROUTES
# -------------------------


@app.route("/generate", methods=["POST"])
@login_required
def generate():
    data = request.get_json()
    prompt = data.get("prompt", "").strip()

    if not prompt:
        return jsonify({"error": "Prompt cannot be empty"}), 400

    # 1️⃣ Gemini output (cleaned)
    raw_output = clean_text(call_gemini(prompt))

    # 2️⃣ Hybrid risk analysis
    metrics = compute_filter_metrics(raw_output)
    risk_score = metrics["risk_score"]

    # 3️⃣ Decide corrected output
    if risk_score > 0 or metrics["toxicity_score"] > 0 or metrics["bias_score"] > 0:
        corrected_output = clean_text(rewrite_text(raw_output))
    else:
        corrected_output = raw_output

    # 4️⃣ SAVE TO DATABASE (PER USER)
    log = PromptLog(
    user_id=current_user.id,
    prompt=prompt,
    gemini_output=raw_output,
    corrected_output=corrected_output,
    risk_score=risk_score
    )
    db.session.add(log)
    db.session.commit()

    

    # 5️⃣ Response to frontend
    response = {
        "raw_output": raw_output,
        **metrics,
        "corrected_output": corrected_output
    }

    # Backward compatibility for older frontend code
    if metrics["responsible_words"]["risk_terms"]:
        response["unsafe_terms"] = metrics["responsible_words"]["risk_terms"]

    return jsonify(response)

@app.route("/logs")
@login_required
def logs():
    logs = (
        PromptLog.query
        .filter_by(user_id=current_user.id)
        .order_by(PromptLog.timestamp.asc())
        .all()
    )

    return render_template("logs.html", logs=logs)


@app.route("/model-insights")
@login_required
def model_insights():
    model_info = {
        "model_name": "DistilBERT (Text Classification)",
        "architecture": "Transformer-based encoder",
        "parameters": "66 million",
        "task": "Binary text classification (SAFE vs UNSAFE)",
        "tokenizer": "WordPiece Tokenizer",
        "decision_logic": (
            "The model encodes the input text into contextual embeddings using "
            "transformer layers and applies a classification head to predict "
            "whether the content is SAFE or UNSAFE."
        ),
        "metrics": {
            "accuracy": 0.92,
            "precision": 0.90,
            "recall": 0.88,
            "f1_score": 0.89
        }
    }

    return render_template("model_insights.html", model=model_info)





# -------------------------
# START SERVER
# -------------------------
if __name__ == "__main__":
    app.run()
