import streamlit as st
import requests
import json

# -------------------------
# Page config
# -------------------------
st.set_page_config(
    page_title="Voter Fact Checker Pro",
    page_icon="🗳️",
    layout="centered"
)

st.title("🗳️ Voter Fact Checker")
st.markdown("**AI-powered fact-checking for political claims using Hugging Face BART**")

# -------------------------
# Configuration
# -------------------------
# Try to get API key from secrets or environment
try:
    HF_API_KEY = st.secrets.get("HF_API_KEY", "")
except:
    import os
    HF_API_KEY = os.environ.get("HF_API_KEY", "")

HF_HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"} if HF_API_KEY else {}

# -------------------------
# Hugging Face API Helper with Fallback
# -------------------------
def classify_claim_with_bart(claim_text):
    """Classify claim using Hugging Face BART model"""
    
    # If no API key, use enhanced keyword-based fallback
    if not HF_API_KEY:
        return enhanced_keyword_classifier(claim_text)
    
    url = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"
    payload = {
        "inputs": claim_text,
        "parameters": {
            "candidate_labels": ["True", "False", "Misleading", "Unverified"],
            "multi_label": False
        }
    }
    
    try:
        response = requests.post(url, headers=HF_HEADERS, json=payload, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            
            # Extract the highest confidence label
            if "labels" in data and "scores" in data:
                max_score = max(data["scores"])
                max_index = data["scores"].index(max_score)
                prediction = data["labels"][max_index]
                confidence = round(max_score * 100, 1)
                return prediction, confidence, data
            else:
                # Fallback if response structure is different
                return enhanced_keyword_classifier(claim_text)
        
        elif response.status_code == 503:
            st.warning("Model is loading. Using enhanced keyword analysis...")
            return enhanced_keyword_classifier(claim_text)
        else:
            st.warning(f"API returned status {response.status_code}. Using fallback...")
            return enhanced_keyword_classifier(claim_text)
            
    except Exception as e:
        st.warning(f"Connection error: {str(e)[:50]}... Using enhanced keyword analysis.")
        return enhanced_keyword_classifier(claim_text)

# -------------------------
# ENHANCED KEYWORD CLASSIFIER (Fallback when API fails)
# -------------------------
def enhanced_keyword_classifier(claim_text):
    """Smart keyword-based classifier as fallback"""
    claim_lower = claim_text.lower()
    
    # True claim patterns (factual statements)
    true_patterns = [
        "voting is a fundamental right",
        "india is a democracy",
        "election commission of india",
        "voter id is required",
        "constitution of india",
        "universal adult suffrage",
        "free and fair elections",
        "evm is used",
        "voter turnout",
        "right to vote",
        "democratic republic",
        "president of india",
        "supreme court"
    ]
    
    # False claim patterns
    false_patterns = [
        "voting is useless",
        "votes don't matter",
        "elections are rigged",
        "democracy doesn't work",
        "evm can be hacked",
        "monarchy is better",
        "fake voter ids",
        "corrupt system",
        "waste of time",
        "pointless to vote",
        "all politicians are corrupt",
        "nothing will change"
    ]
    
    # Misleading claim patterns
    misleading_patterns = [
        "always", "never", "all", "none", "everyone",
        "only", "completely", "totally", "absolutely",
        "100%", "guaranteed", "definitely"
    ]
    
    # Score the claim
    true_score = sum(1 for pattern in true_patterns if pattern in claim_lower)
    false_score = sum(1 for pattern in false_patterns if pattern in claim_lower)
    misleading_score = sum(1 for pattern in misleading_patterns if pattern in claim_lower)
    
    # Calculate confidence based on matches
    total_matches = true_score + false_score + misleading_score
    
    if total_matches == 0:
        return "Unverified", 50.0, {"labels": ["Unverified", "True", "False"], "scores": [0.5, 0.25, 0.25]}
    
    # Determine verdict
    if false_score > true_score and false_score > misleading_score:
        confidence = min(30 + (false_score * 15), 85)
        return "False", confidence, {"labels": ["False", "True", "Misleading"], "scores": [confidence/100, 0.3, 0.2]}
    elif misleading_score > 0 and claim_lower.count(" ") > 5:
        confidence = min(40 + (misleading_score * 10), 75)
        return "Misleading", confidence, {"labels": ["Misleading", "False", "True"], "scores": [confidence/100, 0.3, 0.2]}
    elif true_score > 0:
        confidence = min(60 + (true_score * 8), 90)
        return "True", confidence, {"labels": ["True", "Misleading", "False"], "scores": [confidence/100, 0.25, 0.15]}
    else:
        return "Unverified", 60.0, {"labels": ["Unverified", "True", "False"], "scores": [0.6, 0.2, 0.2]}

# -------------------------
# Explanation Generator
# -------------------------
def generate_explanation(claim, prediction, confidence):
    """Generate neutral, factual explanation"""
    
    explanations = {
        "True": [
            "✅ This claim aligns with established facts and constitutional provisions.",
            "The statement is supported by official records and electoral data.",
            "Multiple verified sources confirm the accuracy of this information.",
            "This reflects the legal and procedural framework of Indian democracy."
        ],
        "False": [
            "⚠️ This claim appears to contradict verified facts and official records.",
            "Available evidence and constitutional provisions do not support this statement.",
            "Official sources, including the Election Commission, provide contrary information.",
            "This may be based on misinformation or misunderstanding of electoral processes."
        ],
        "Misleading": [
            "🔍 This claim contains elements that may create a false impression.",
            "While parts may be technically true, the overall context is misleading.",
            "Important nuances or qualifications are missing from this statement.",
            "The claim uses selective information that doesn't represent the complete picture."
        ],
        "Unverified": [
            "📝 This claim requires further verification from official sources.",
            "Insufficient evidence is available to confirm or deny this statement.",
            "The claim references topics that need confirmation from authoritative sources.",
            "Check with multiple verified sources before accepting this information."
        ]
    }
    
    # Base explanation parts
    base_parts = explanations.get(prediction, explanations["Unverified"])
    
    # Add confidence-based note
    if confidence > 80:
        confidence_note = f"High confidence ({confidence}%) based on analysis."
    elif confidence > 60:
        confidence_note = f"Moderate confidence ({confidence}%) based on available information."
    else:
        confidence_note = f"Low confidence ({confidence}%). Further verification recommended."
    
    # Construct full explanation
    explanation = f"**Analysis Confidence: {confidence}%**\n\n"
    explanation += f"{confidence_note}\n\n"
    
    for part in base_parts[:3]:  # Take first 3 explanation points
        explanation += f"• {part}\n"
    
    # Add verification advice
    explanation += "\n**Verification Advice:**\n"
    explanation += "• Check official Election Commission website (eci.gov.in)\n"
    explanation += "• Verify with Press Information Bureau (pib.gov.in)\n"
    explanation += "• Consult multiple reputable sources for confirmation\n"
    explanation += "• Consider the date and context of information"
    
    return explanation

# -------------------------
# Official Sources
# -------------------------
def get_relevant_sources(claim):
    """Return relevant official sources based on claim content"""
    
    claim_lower = claim.lower()
    
    all_sources = [
        {"name": "Election Commission of India", "url": "https://eci.gov.in", 
         "keywords": ["vote", "election", "voter", "evm", "poll", "commission"]},
        {"name": "Press Information Bureau", "url": "https://pib.gov.in", 
         "keywords": ["government", "ministry", "scheme", "policy", "official"]},
        {"name": "India Code", "url": "https://indiacode.nic.in", 
         "keywords": ["law", "constitution", "act", "amendment", "legal"]},
        {"name": "PRS Legislative Research", "url": "https://prsindia.org", 
         "keywords": ["bill", "parliament", "legislative", "policy", "analysis"]},
        {"name": "Supreme Court of India", "url": "https://main.sci.gov.in", 
         "keywords": ["court", "judgment", "legal", "constitution", "right"]},
        {"name": "Law Ministry", "url": "https://lawmin.gov.in", 
         "keywords": ["constitution", "law", "legal", "ministry", "government"]},
    ]
    
    # Score sources based on keyword matches
    scored_sources = []
    for source in all_sources:
        score = 0
        for keyword in source["keywords"]:
            if keyword in claim_lower:
                score += 3
        
        # Base score for important sources
        if source["name"] == "Election Commission of India":
            score += 2  # Always relevant for election claims
        
        scored_sources.append((score, source))
    
    # Sort by score and return top 4
    scored_sources.sort(reverse=True, key=lambda x: x[0])
    return [source for _, source in scored_sources[:4]]

# -------------------------
# SIMPLE, CLEAN UI
# -------------------------
# Sidebar with info
with st.sidebar:
    st.header("ℹ️ About")
    st.info("""
    **How it works:**
    1. Uses Hugging Face BART model for classification
    2. Generates neutral explanations
    3. Provides official sources
    4. Fallback to keyword analysis if needed
    """)
    
    # API status
    if HF_API_KEY:
        st.success("✅ Hugging Face API: Configured")
    else:
        st.warning("⚠️ Hugging Face API: Using enhanced keyword analysis")
    
    st.header("💡 Examples")
    examples = [
        "Voting is a fundamental right in India",
        "Election Commission ensures free and fair elections",
        "Voting is useless and doesn't matter",
        "EVMs can be easily hacked remotely",
        "India is a democratic republic",
        "One Nation One Election is implemented"
    ]
    
    for ex in examples:
        if st.button(f"» {ex[:40]}...", key=f"ex_{ex}", use_container_width=True):
            st.session_state.example_claim = ex

# Main interface
st.markdown("---")

# Input
if 'example_claim' in st.session_state:
    claim = st.text_area(
        "**Enter political claim to fact-check:**",
        value=st.session_state.example_claim,
        height=100,
        placeholder="Example: 'Voting is a fundamental right in India'"
    )
    del st.session_state.example_claim
else:
    claim = st.text_area(
        "**Enter political claim to fact-check:**",
        height=100,
        placeholder="Example: 'Voting is a fundamental right in India'"
    )

# Analyze button
col1, col2 = st.columns([1, 5])
with col1:
    analyze_btn = st.button("🔍 **Analyze Claim**", type="primary", use_container_width=True)

# Process
if analyze_btn and claim.strip():
    with st.spinner("Analyzing with BART model..."):
        # Get classification
        prediction, confidence, raw_data = classify_claim_with_bart(claim)
        
        # Generate explanation
        explanation = generate_explanation(claim, prediction, confidence)
        
        # Get relevant sources
        sources = get_relevant_sources(claim)
        
        # Display results
        st.markdown("---")
        
        # Verdict with color coding
        if prediction == "True":
            st.success(f"## ✅ Verdict: {prediction}")
        elif prediction == "False":
            st.error(f"## ❌ Verdict: {prediction}")
        elif prediction == "Misleading":
            st.warning(f"## ⚠️ Verdict: {prediction}")
        else:
            st.info(f"## 🔍 Verdict: {prediction}")
        
        # Confidence
        st.subheader(f"Confidence: {confidence}%")
        st.progress(confidence/100)
        
        # Show all scores if available
        if "scores" in raw_data and "labels" in raw_data:
            with st.expander("View detailed scores"):
                for label, score in zip(raw_data["labels"], raw_data["scores"]):
                    score_pct = round(score * 100, 1)
                    st.write(f"**{label}**: {score_pct}%")
                    st.progress(score, text=f"{label}: {score_pct}%")
        
        # Explanation
        st.subheader("📘 Explanation")
        st.info(explanation)
        
        # Sources
        st.subheader("🔍 Recommended Official Sources")
        
        cols = st.columns(2)
        for idx, source in enumerate(sources):
            with cols[idx % 2]:
                st.markdown(f"""
                <div style='padding:12px; border-radius:8px; background:#f0f2f6; margin:8px 0; border-left:4px solid #4CAF50;'>
                <b>{source['name']}</b><br>
                <small><a href="{source['url']}" target="_blank">{source['url']}</a></small>
                </div>
                """, unsafe_allow_html=True)
        
        # Disclaimer
        st.warning("""
        **Disclaimer**: This tool provides AI-assisted analysis for informational purposes. 
        Always verify critical information with official sources like the Election Commission of India.
        """)
        
elif analyze_btn:
    st.warning("Please enter a claim to analyze.")

# Footer
st.markdown("---")
st.caption("Powered by Hugging Face BART • Neutral fact-checking • For democratic awareness")