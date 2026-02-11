import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)

# ---------------------------
# 1. LOAD DATASET
# ---------------------------

df = pd.read_csv("output_filter_dataset.csv")

print("Dataset size:", df.shape)
print(df["label"].value_counts())

# ---------------------------
# 2. PREPROCESSING
# ---------------------------

# Clean text (basic but effective)
df["text"] = df["text"].str.lower().str.strip()

# Encode labels
label_map = {"SAFE": 0, "UNSAFE": 1}
df["label"] = df["label"].map(label_map)

# ---------------------------
# 3. TRAIN / VALIDATION SPLIT
# ---------------------------

X_train, X_test, y_train, y_test = train_test_split(
    df["text"].tolist(),
    df["label"].tolist(),
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

# ---------------------------
# 4. TOKENIZATION
# ---------------------------

tokenizer = DistilBertTokenizerFast.from_pretrained(
    "distilbert-base-uncased"
)

train_encodings = tokenizer(
    X_train, truncation=True, padding=True
)
test_encodings = tokenizer(
    X_test, truncation=True, padding=True
)

# ---------------------------
# 5. DATASET CLASS
# ---------------------------

class FilterDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = FilterDataset(train_encodings, y_train)
test_dataset = FilterDataset(test_encodings, y_test)

# ---------------------------
# 6. LOAD MODEL
# ---------------------------

model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
)

# ---------------------------
# 7. TRAINING CONFIG
# ---------------------------

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_dir="./logs",
    logging_steps=50,
    optim="adamw_torch",   # 🔥 THIS LINE FIXES IT
    save_safetensors=False 
)


# ---------------------------
# 8. METRICS FUNCTION
# ---------------------------

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary"
    )
    acc = accuracy_score(labels, predictions)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

# ---------------------------
# 9. TRAINER
# ---------------------------

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# ---------------------------
# 10. TRAIN MODEL
# ---------------------------

trainer.train()

# ---------------------------
# 11. EVALUATION
# ---------------------------

predictions = trainer.predict(test_dataset)
y_pred = np.argmax(predictions.predictions, axis=1)

acc = accuracy_score(y_test, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(
    y_test, y_pred, average="binary"
)

print("\n--- FINAL EVALUATION ---")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

# ---------------------------
# 12. CONFUSION MATRIX
# ---------------------------

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5,4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["SAFE", "UNSAFE"],
    yticklabels=["SAFE", "UNSAFE"]
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# ---------------------------
# 13. SAVE MODEL
# ---------------------------

model.save_pretrained("saved_filter_model")
tokenizer.save_pretrained("saved_filter_model")

print("\n✅ Model saved in 'saved_filter_model/'")