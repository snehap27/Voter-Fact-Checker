import streamlit as st
import requests
from serpapi import GoogleSearch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import time

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
# Configuration & Secrets
# -------------------------
# Get API keys from secrets or environment variables
try:
    HF_API_KEY = st.secrets.get("HF_API_KEY", os.getenv("HF_API_KEY", ""))
    SERP_API_KEY = st.secrets.get("SERP_API_KEY", os.getenv("SERP_API_KEY", ""))
except:
    HF_API_KEY = os.getenv("HF_API_KEY", "")
    SERP_API_KEY = os.getenv("SERP_API_KEY", "")

HF_HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"} if HF_API_KEY else {}

# -------------------------
# HF API helpers with error handling
# -------------------------
@st.cache_data(show_spinner=False, ttl=3600)
def hf_zero_shot(text, labels, max_retries=3):
    """Classify claim using zero-shot classification"""
    if not HF_API_KEY:
        st.error("⚠️ Hugging Face API key not configured. Please add HF_API_KEY to secrets.")
        return None
    
    url = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"
    payload = {
        "inputs": text,
        "parameters": {"candidate_labels": labels}
    }
    
    for attempt in range(max_retries):
        try:
            r = requests.post(url, headers=HF_HEADERS, json=payload, timeout=30)
            if r.status_code == 200:
                return r.json()
            elif r.status_code == 503:
                # Model is loading, wait and retry
                time.sleep(2 ** attempt)  # Exponential backoff
                continue
            else:
                st.error(f"API Error: {r.status_code}")
                return None
        except requests.exceptions.Timeout:
            if attempt == max_retries - 1:
                st.error("Request timed out. Please try again.")
                return None
            time.sleep(1)
        except Exception as e:
            st.error(f"Error: {str(e)}")
            return None
    return None

@st.cache_data(show_spinner=False, ttl=3600)
def hf_explain(prompt, max_retries=2):
    """Generate explanation for the claim"""
    if not HF_API_KEY:
        return "Explanation unavailable (API key not configured)."
    
    url = "https://api-inference.huggingface.co/models/google/flan-t5-base"
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 200,
            "temperature": 0.7,
            "do_sample": True
        }
    }
    
    for attempt in range(max_retries):
        try:
            r = requests.post(url, headers=HF_HEADERS, json=payload, timeout=30)
            if r.status_code == 200:
                out = r.json()
                if isinstance(out, list) and len(out) > 0:
                    return out[0].get("generated_text", "Explanation unavailable.")
            time.sleep(1)
        except:
            time.sleep(1)
    
    return "Explanation temporarily unavailable. Please try again."

# -------------------------
# Trusted sources
# -------------------------
TRUSTED_SITES = [
    "site:eci.gov.in",
    "site:prsindia.org",
    "site:legislative.gov.in",
    "site:indiacode.nic.in",
    "site:scobserver.in",
    "site:pib.gov.in",  # Added Press Information Bureau
    "site:indianexpress.com",  # Added reputed news source
    "site:thehindu.com"  # Added reputed news source
]

# -------------------------
# Explanation generator
# -------------------------
def explain_claim(claim, prediction):
    """Generate context-aware explanation"""
    prompt = f"""
You are a neutral fact-checking assistant for Indian elections and governance.

Claim: "{claim}"
Classification: {prediction}

Provide a balanced explanation considering:
1. Legal/constitutional context if relevant
2. Procedural aspects
3. Historical precedents if any
4. Why this might be classified as {prediction}
Keep it factual, neutral, and educational (3-4 sentences).

Explanation:
"""
    return hf_explain(prompt)

# -------------------------
# Source search
# -------------------------
def extract_search_terms(claim):
    """Extract relevant search terms from claim"""
    claim_lower = claim.lower()
    
    # Add more keywords as needed
    keywords = {
        "election": ["election", "vote", "polling", "voting"],
        "constitution": ["constitution", "article", "amendment", "fundamental"],
        "government": ["government", "ministry", "parliament", "lok sabha"],
        "policy": ["policy", "scheme", "program", "yojana"]
    }
    
    search_terms = []
    for category, terms in keywords.items():
        if any(term in claim_lower for term in terms):
            search_terms.append(category)
    
    if not search_terms:
        search_terms = ["Indian governance"]
    
    return " ".join(search_terms)

@st.cache_data(show_spinner=False, ttl=1800)  # Cache for 30 minutes
def fetch_sources(claim):
    """Fetch sources from trusted sites"""
    if not SERP_API_KEY:
        st.warning("🔍 Search API key not configured. Using sample sources.")
        return get_sample_sources(claim)
    
    query = f"{extract_search_terms(claim)} {' OR '.join(TRUSTED_SITES[:5])}"
    
    try:
        params = {
            "q": query,
            "hl": "en",
            "num": 8,
            "api_key": SERP_API_KEY,
            "engine": "google"
        }
        
        search = GoogleSearch(params)
        results = search.get_dict()
        
        if "error" in results:
            st.warning("Search API limit reached. Using sample sources.")
            return get_sample_sources(claim)
        
        links = []
        for r in results.get("organic_results", [])[:6]:
            link = {
                "title": r.get("title", "No title"),
                "link": r.get("link", "#"),
                "snippet": r.get("snippet", "No description available")
            }
            links.append(link)
        
        return rank_sources(claim, links)
        
    except Exception as e:
        st.warning(f"Search failed: {str(e)}. Using sample sources.")
        return get_sample_sources(claim)

def get_sample_sources(claim):
    """Return sample sources when API is unavailable"""
    sample_sources = [
        {
            "title": "Election Commission of India - Official Website",
            "link": "https://eci.gov.in",
            "snippet": "Official source for election-related information in India"
        },
        {
            "title": "PRS Legislative Research",
            "link": "https://prsindia.org",
            "snippet": "Independent research on legislation and governance"
        },
        {
            "title": "India Code - Digital Repository of Laws",
            "link": "https://www.indiacode.nic.in",
            "snippet": "Official database of central and state laws"
        }
    ]
    return sample_sources

def rank_sources(claim, links):
    """Rank sources by relevance to claim"""
    if not links or len(links) < 2:
        return links
    
    try:
        texts = [claim] + [f"{l['title']} {l['snippet']}" for l in links]
        vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
        tfidf_matrix = vectorizer.fit_transform(texts)
        
        similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
        
        ranked = sorted(zip(similarities, links), key=lambda x: x[0], reverse=True)
        return [link for _, link in ranked[:3]]
    except:
        return links[:3]  # Fallback to first 3 if ranking fails

# -------------------------
# Main UI
# -------------------------
st.markdown("---")

# Sidebar with instructions
with st.sidebar:
    st.header("ℹ️ How to Use")
    st.markdown("""
    1. **Enter a claim** about Indian politics/elections
    2. Click **"Check Fact"** to analyze
    3. Review the **analysis result**
    4. Check **explanations** and **sources**
    5. Always verify with official sources
    
    **Note:** This tool is for educational purposes only.
    """)
    
    st.header("⚙️ Configuration Status")
    if HF_API_KEY:
        st.success("✅ Hugging Face API: Configured")
    else:
        st.error("❌ Hugging Face API: Missing")
        
    if SERP_API_KEY:
        st.success("✅ Search API: Configured")
    else:
        st.warning("⚠️ Search API: Missing (using sample sources)")
    
    st.markdown("---")
    st.caption("Made with ❤️ for democratic awareness")

# Main input area
claim = st.text_area(
    "**Enter a political or election-related claim:**",
    placeholder="Example: One Nation One Election requires constitutional amendment",
    height=100
)

col1, col2 = st.columns([1, 3])
with col1:
    check_button = st.button("🔍 Check Fact", type="primary", use_container_width=True)
with col2:
    if st.button("📋 Example Claims", use_container_width=True):
        st.session_state.example_shown = True

if 'example_shown' in st.session_state and st.session_state.example_shown:
    with st.expander("💡 Try these example claims:"):
        examples = [
            "One Nation One Election will save ₹10,000 crore",
            "Voter ID cards can be downloaded from WhatsApp",
            "Electronic voting machines can be hacked remotely",
            "The Constitution allows simultaneous elections",
            "Voting is mandatory in all elections"
        ]
        for example in examples:
            if st.button(example, key=example):
                st.session_state.claim_text = example
                st.rerun()

if check_button or ('claim_text' in st.session_state and st.session_state.claim_text):
    claim_to_check = st.session_state.get('claim_text', claim) if 'claim_text' in st.session_state else claim
    
    if not claim_to_check.strip():
        st.warning("Please enter a claim to check.")
    else:
        with st.spinner("Analyzing claim..."):
            # Zero-shot classification
            labels = ["True", "False", "Misleading", "Unverified"]
            result = hf_zero_shot(claim_to_check, labels)
            
            if result:
                prediction = result["labels"][0]
                confidence = round(result["scores"][0] * 100, 2)
                
                # Display result with color coding
                st.subheader("📊 Analysis Result")
                
                col1, col2 = st.columns(2)
                with col1:
                    # Color code based on prediction
                    if prediction == "True":
                        st.markdown(f"**Prediction:** ✅ {prediction}")
                    elif prediction == "False":
                        st.markdown(f"**Prediction:** ❌ {prediction}")
                    elif prediction == "Misleading":
                        st.markdown(f"**Prediction:** ⚠️ {prediction}")
                    else:
                        st.markdown(f"**Prediction:** 🔍 {prediction}")
                
                with col2:
                    # Confidence indicator
                    if confidence > 70:
                        color = "green"
                    elif confidence > 40:
                        color = "orange"
                    else:
                        color = "red"
                    st.markdown(f"**Confidence:** <span style='color:{color}'>{confidence}%</span>", 
                               unsafe_allow_html=True)
                
                # Show all scores
                with st.expander("View detailed scores"):
                    for label, score in zip(result["labels"], result["scores"]):
                        score_pct = round(score * 100, 1)
                        st.progress(score, text=f"{label}: {score_pct}%")
                
                # Explanation
                st.subheader("📘 Explanation")
                explanation = explain_claim(claim_to_check, prediction)
                st.info(explanation)
                
                # Sources
                st.subheader("🔍 Recommended Sources")
                with st.spinner("Searching trusted sources..."):
                    sources = fetch_sources(claim_to_check)
                    
                if sources:
                    for i, source in enumerate(sources, 1):
                        with st.container():
                            st.markdown(f"**{i}. [{source['title']}]({source['link']})**")
                            st.caption(source['snippet'])
                            st.markdown("---")
                else:
                    st.warning("No sources found. Please check the claim wording.")
                
                # Disclaimer
                st.info("""
                ⚠️ **Disclaimer:** This analysis is AI-generated and should be used for informational purposes only. 
                Always verify information with official sources before making decisions.
                """)
                
                # Reset example claim
                if 'claim_text' in st.session_state:
                    del st.session_state.claim_text
            else:
                st.error("Unable to analyze the claim. Please try again or check your API configuration.")

st.markdown("---")
st.caption("""
Built for voter awareness and democratic education. 
This platform does not endorse any political party or ideology.
""")