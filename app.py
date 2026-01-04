import streamlit as st
import requests
from serpapi import GoogleSearch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------
# Page config
# -------------------------
st.set_page_config(
    page_title="Voter Fact Checker",
    page_icon="🗳️",
    layout="centered"
)

st.title("🗳️ Voter Awareness & Fact Checker")
st.write(
    "This tool analyzes political claims in a neutral, non-partisan way. "
    "It does not support or oppose any political ideology."
)

HF_API_KEY = st.secrets["HF_API_KEY"]
HF_HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"}

# -------------------------
# HF API helpers
# -------------------------
def hf_zero_shot(text, labels):
    url = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"
    payload = {
        "inputs": text,
        "parameters": {"candidate_labels": labels}
    }
    r = requests.post(url, headers=HF_HEADERS, json=payload, timeout=60)
    return r.json()

def hf_explain(prompt):
    url = "https://api-inference.huggingface.co/models/google/flan-t5-base"
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 200}
    }
    r = requests.post(url, headers=HF_HEADERS, json=payload, timeout=60)
    out = r.json()
    return out[0]["generated_text"] if isinstance(out, list) else "Explanation unavailable."

# -------------------------
# Trusted sources
# -------------------------
TRUSTED_SITES = [
    "site:eci.gov.in",
    "site:prsindia.org",
    "site:legislative.gov.in",
    "site:indiacode.nic.in",
    "site:scobserver.in"
]

# -------------------------
# Explanation
# -------------------------
def explain_claim(claim, prediction):
    prompt = f"""
You are a neutral fact-checking assistant.

Claim classification: {prediction}

Explain why the claim may be misleading.
Do not repeat the claim.
Mention legal, constitutional, or procedural context.
Write 4 clear sentences.
"""
    return hf_explain(prompt)

# -------------------------
# Source search
# -------------------------
def extract_search_terms(claim):
    if "one nation one election" in claim.lower():
        return "One Nation One Election constitutional amendment"
    return "Indian election reform law"

def fetch_sources(claim):
    if "SERP_API_KEY" not in st.secrets:
        return []

    query = f"{extract_search_terms(claim)} {' OR '.join(TRUSTED_SITES)}"
    params = {
        "q": query,
        "hl": "en",
        "num": 6,
        "api_key": st.secrets["SERP_API_KEY"]
    }

    search = GoogleSearch(params)
    results = search.get_dict()

    links = [{
        "title": r.get("title", ""),
        "link": r.get("link", ""),
        "snippet": r.get("snippet", "")
    } for r in results.get("organic_results", [])]

    return rank_sources(claim, links)

def rank_sources(claim, links):
    if not links:
        return []

    texts = [claim] + [l["title"] + " " + l["snippet"] for l in links]
    tfidf = TfidfVectorizer(stop_words="english").fit_transform(texts)
    sims = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()

    ranked = sorted(zip(sims, links), reverse=True)
    return [l for _, l in ranked[:3]]

# -------------------------
# UI
# -------------------------
claim = st.text_area(
    "Enter a political or election-related claim:",
    placeholder="Example: One Nation One Election will destroy democracy"
)

if st.button("Check Fact"):
    if not claim.strip():
        st.warning("Please enter a claim.")
    else:
        labels = ["True", "False", "Misleading"]
        result = hf_zero_shot(claim, labels)

        prediction = result["labels"][0]
        confidence = round(result["scores"][0] * 100, 2)

        st.subheader("🧠 Analysis Result")
        st.write(f"**Prediction:** {prediction}")
        st.write(f"**Confidence:** {confidence}%")

        st.subheader("📘 Explanation")
        st.write(explain_claim(claim, prediction))

        st.subheader("🔍 Official Sources")
        sources = fetch_sources(claim)
        if sources:
            for s in sources:
                st.markdown(f"- [{s['title']}]({s['link']})")
        else:
            st.write("No official sources retrieved automatically.")

        st.info(
            "⚠️ This platform does not tell users what to believe. "
            "It only checks factual verifiability using public records."
        )

st.markdown("---")
st.caption("Built for voter awareness and democratic education.")
