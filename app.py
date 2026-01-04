import streamlit as st
from transformers import pipeline
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

# -------------------------
# Load models
# -------------------------
@st.cache_resource
def load_classifier():
    return pipeline(
        "zero-shot-classification",
        model="typeform/distilbert-base-uncased-mnli"
    )

@st.cache_resource
def load_reasoner():
    return pipeline(
        "text2text-generation",
        model="google/flan-t5-base"
    )

classifier = load_classifier()
reasoner = load_reasoner()

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
# Explanation (ROBUST)
# -------------------------
def explain_claim(claim, prediction):
    prompt = (
        "You are a neutral fact-checking assistant.\n\n"
        f"Claim: {claim}\n\n"
        f"Classification: {prediction}\n\n"
        "Explain why this claim may be misleading.\n"
        "Do NOT repeat the claim.\n"
        "Mention missing context, legal or constitutional process, "
        "and why the conclusion is oversimplified.\n"
        "Write 4 complete sentences.\n\n"
        "Explanation:"
    )

    output = reasoner(
        prompt,
        max_length=220,
        do_sample=True,
        temperature=0.7
    )

    text = output[0]["generated_text"].strip()

    # Fallback if model fails
    if claim.lower() in text.lower() or len(text.split()) < 20:
        return (
            "The claim draws a strong conclusion without accounting for how election "
            "reforms are implemented in practice. In India, any change to the election "
            "schedule would require constitutional amendments, parliamentary approval, "
            "and judicial oversight. Democratic functioning depends on multiple institutions "
            "such as the legislature, courts, and independent bodies, not only on election timing. "
            "The claim oversimplifies a complex policy discussion by ignoring these safeguards."
        )

    return text

# -------------------------
# Keyword extraction
# -------------------------
def extract_search_terms(claim):
    if "one nation one election" in claim.lower():
        return "One Nation One Election"
    if "simultaneous" in claim.lower():
        return "simultaneous elections India"
    return "election reform India"

# -------------------------
# Fetch sources
# -------------------------
def fetch_sources(claim):
    if "SERP_API_KEY" not in st.secrets:
        return []

    search_term = extract_search_terms(claim)

    query = (
        f"{search_term} constitutional amendment "
        f"{' OR '.join(TRUSTED_SITES)}"
    )

    params = {
        "q": query,
        "hl": "en",
        "num": 6,
        "api_key": st.secrets["SERP_API_KEY"]
    }

    search = GoogleSearch(params)
    results = search.get_dict()

    links = []
    for r in results.get("organic_results", []):
        links.append({
            "title": r.get("title", ""),
            "link": r.get("link", ""),
            "snippet": r.get("snippet", "")
        })

    return rank_sources(claim, links)

# -------------------------
# Rank sources (NO OVER-FILTERING)
# -------------------------
def rank_sources(claim, links):
    if not links:
        return []

    texts = [claim] + [
        l["title"] + " " + l["snippet"] for l in links
    ]

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2)
    )

    tfidf = vectorizer.fit_transform(texts)
    similarities = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()

    ranked = list(zip(similarities, links))
    ranked.sort(reverse=True, key=lambda x: x[0])

    # Return top matches (even if score is low)
    return [l for _, l in ranked[:3]]

# -------------------------
# User input
# -------------------------
claim = st.text_area(
    "Enter a political or election-related claim:",
    placeholder="Example: One Nation One Election will destroy democracy"
)

# -------------------------
# Fact check
# -------------------------
if st.button("Check Fact"):
    if not claim.strip():
        st.warning("Please enter a claim.")
    else:
        labels = ["True", "False", "Misleading"]
        result = classifier(claim, labels)

        prediction = result["labels"][0]
        confidence = round(result["scores"][0] * 100, 2)

        st.subheader("🧠 Analysis Result")
        st.write(f"**Prediction:** {prediction}")
        st.write(f"**Confidence:** {confidence}%")

        st.subheader("📘 Why this claim may be misleading")
        st.write(explain_claim(claim, prediction))

        st.subheader("🔍 What voters should verify")
        sources = fetch_sources(claim)

        if sources:
            for s in sources:
                st.markdown(f"- [{s['title']}]({s['link']})")
        else:
            st.write(
                "No official sources could be retrieved automatically. "
                "Users are encouraged to verify the claim using Election Commission "
                "and parliamentary records."
            )

        st.info(
            "⚠️ This tool does not declare absolute truth. "
            "It encourages voters to verify claims using official public sources."
        )

# -------------------------
# Footer
# -------------------------
st.markdown("---")
st.caption("Built for voter awareness and democratic education.")
