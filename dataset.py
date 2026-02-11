import csv
import random

safe_openers = [
    "This clause",
    "This section",
    "The provision",
    "This part of the agreement",
    "The contract language",
    "This paragraph",
    "The stated condition"
]

safe_verbs = [
    "explains",
    "describes",
    "outlines",
    "mentions",
    "summarizes",
    "clarifies",
    "indicates"
]

safe_modifiers = [
    "in general terms",
    "without offering guarantees",
    "in a neutral manner",
    "for informational purposes",
    "without legal certainty",
    "in plain language",
    "at a high level"
]

safe_topics = [
    "payment obligations",
    "termination conditions",
    "confidentiality requirements",
    "liability limitations",
    "contract duration",
    "financial responsibilities",
    "dispute resolution process",
    "service scope",
    "penalty clauses",
    "risk allocation"
]

unsafe_claims = [
    "guarantees complete protection",
    "ensures there is no legal risk",
    "makes the agreement completely safe",
    "removes all liability",
    "guarantees success in court",
    "ensures permanent employment",
    "eliminates the need for legal advice",
    "provides 100 percent safety",
    "cannot be challenged legally",
    "ensures you will always win"
]

advice_phrases = [
    "you can safely sign this",
    "there is absolutely no risk",
    "this is legally foolproof",
    "you do not need a lawyer",
    "this agreement is perfectly secure",
    "there is nothing to worry about"
]

rows = []

# SAFE samples
for _ in range(550):
    sentence = (
        f"{random.choice(safe_openers)} "
        f"{random.choice(safe_verbs)} "
        f"the {random.choice(safe_topics)} "
        f"{random.choice(safe_modifiers)}."
    )
    rows.append([sentence, "SAFE"])

# UNSAFE samples
for _ in range(550):
    if random.random() > 0.5:
        sentence = (
            f"This clause {random.choice(unsafe_claims)} "
            f"related to {random.choice(safe_topics)}."
        )
    else:
        sentence = (
            f"{random.choice(advice_phrases).capitalize()} "
            f"because this clause {random.choice(unsafe_claims)}."
        )
    rows.append([sentence, "UNSAFE"])

# Shuffle dataset
random.shuffle(rows)

# Write CSV
with open("output_filter_dataset.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["text", "label"])
    writer.writerows(rows)

print(f"✅ Dataset generated with {len(rows)} diverse rows.")
