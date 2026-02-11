function generate() {
    const prompt = document.getElementById("prompt").value;

    fetch("/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt })
    })
    .then(res => res.json())
    .then(data => {
        document.getElementById("raw_output").innerHTML =
    formatText(data.raw_output);
        document.getElementById("filter_result").innerHTML =
    renderFilterResult(data);
        document.getElementById("corrected_output").innerHTML =
    formatText(data.corrected_output);
    });
}

function escapeHtml(value) {
    return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}

function renderTerms(title, termsObj) {
    const entries = Object.entries(termsObj || {});
    if (!entries.length) return "";

    const items = entries
        .map(([term, reason]) => `<li><strong>${escapeHtml(term)}:</strong> ${escapeHtml(reason)}</li>`)
        .join("");

    return `
        <div class="terms-block">
            <div class="terms-title"><strong>${escapeHtml(title)}</strong></div>
            <ul class="terms-list">${items}</ul>
        </div>
    `;
}

function renderFilterResult(data) {
    const prediction = data.ml_prediction || data.prediction || "—";
    const riskScore = data.risk_score ?? 0;
    const riskProb = data.risk_probability;
    const toxScore = data.toxicity_score ?? 0;
    const biasScore = data.bias_score ?? 0;

    const responsible = data.responsible_words || {};
    const riskTerms = responsible.risk_terms || data.unsafe_terms || {};
    const toxicityTerms = responsible.toxicity_terms || {};
    const biasTerms = responsible.bias_terms || {};

    const badgeClass = prediction === "SAFE" ? "safe" : (prediction === "UNSAFE" ? "unsafe" : "");
    const riskProbText = (typeof riskProb === "number") ? `${riskProb}%` : "—";

    return `
        <div>
            <strong>Prediction:</strong> <span class="badge ${badgeClass}">${escapeHtml(prediction)}</span>
            <br><strong>Risk Score:</strong> ${escapeHtml(riskScore)}
            <br><strong>Risk Probability:</strong> ${escapeHtml(riskProbText)}
            <br><strong>Toxicity Score:</strong> ${escapeHtml(toxScore)}
            <br><strong>Bias Score:</strong> ${escapeHtml(biasScore)}

            ${renderTerms("Risk terms (responsible words)", riskTerms)}
            ${renderTerms("Toxicity terms (responsible words)", toxicityTerms)}
            ${renderTerms("Bias terms (responsible words)", biasTerms)}
        </div>
    `;
}



function formatText(text) {
    return text
        .split("\n\n")
        .map(p => `<p>${p}</p>`)
        .join("");
}
