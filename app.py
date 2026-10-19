"""
=============================================================================
PROJECT TITLE: AI-Powered FAQ Generator and Chatbot for a College Department
MCA Generative AI Lab Practical Examination
=============================================================================
This application provides a retrieval-grounded FAQ chatbot for a college MCA
department. It uses a structured JSON knowledge base and TF-IDF Cosine Similarity
for matching student queries, with an optional Generative AI (Gemini) RAG layer.
If no API key is supplied, it runs 100% locally and deterministically.
=============================================================================
"""

import os
import json
import re
from typing import Tuple, Dict, Any, Optional

import streamlit as st
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Optional Generative AI library
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False


# =============================================================================
# 1. CONSTANTS & CONFIGURATION
# =============================================================================
FAQ_FILE_PATH = os.path.join(os.path.dirname(__file__), "faq_data.json")
DEFAULT_SIMILARITY_THRESHOLD = 0.22

FALLBACK_RESPONSE = (
    "I am sorry, but I do not have verified information regarding your query "
    "in the departmental FAQ knowledge base.\n\n"
    "📌 **Next Steps:**\n"
    "- Please contact the **MCA Department Office** (Room 204, Academic Block A).\n"
    "- Email: `mca-office@college.edu` | Helpline: `+91-98765-43210`\n"
    "- Office Hours: Monday to Saturday, 9:30 AM to 5:00 PM."
)


# =============================================================================
# 2. CORE RETRIEVAL ENGINE & NLP UTILITIES
# =============================================================================
def normalize_text(text: str) -> str:
    """
    Normalizes input text by lowercasing, stripping special characters,
    and collapsing multiple whitespace characters.
    """
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_faq_database(filepath: str = FAQ_FILE_PATH) -> list:
    """
    Loads and validates the structured FAQ dataset from a JSON file.
    Ensures every entry has required keys: id, category, question, answer, keywords.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Knowledge base file not found at: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("FAQ JSON must contain a root list of FAQ objects.")

    required_keys = {"id", "category", "question", "answer", "keywords"}
    for idx, item in enumerate(data):
        missing = required_keys - set(item.keys())
        if missing:
            raise ValueError(f"FAQ item at index {idx} missing required keys: {missing}")

    return data


class FAQRetriever:
    """
    Retrieval engine utilizing TF-IDF Vectorization and Cosine Similarity
    to match user queries against structured FAQ documents.
    """
    def __init__(self, faq_list: list):
        self.faq_list = faq_list
        self.corpus = []
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english"
        )
        self._prepare_corpus()

    def _prepare_corpus(self):
        """
        Creates combined searchable text for each FAQ entry:
        question + keywords + category.
        """
        self.corpus = []
        for entry in self.faq_list:
            keywords_str = " ".join(entry.get("keywords", []))
            combined_text = f"{entry['question']} {keywords_str} {entry.get('category', '')}"
            self.corpus.append(normalize_text(combined_text))

        self.tfidf_matrix = self.vectorizer.fit_transform(self.corpus)
        # Build vocabulary list for automatic typo-correction (e.g. 'addmission' -> 'admission')
        self.vocabulary_words = list(self.vectorizer.vocabulary_.keys())

    def auto_correct_query(self, query: str) -> str:
        """Corrects common student typos against the known departmental vocabulary."""
        import difflib
        words = query.split()
        corrected = []
        for w in words:
            if len(w) >= 4 and w not in self.vectorizer.vocabulary_:
                matches = difflib.get_close_matches(w, self.vocabulary_words, n=1, cutoff=0.75)
                corrected.append(matches[0] if matches else w)
            else:
                corrected.append(w)
        return " ".join(corrected)

    def retrieve(self, query: str, threshold: float = DEFAULT_SIMILARITY_THRESHOLD) -> Tuple[Optional[Dict[str, Any]], float]:
        """
        Matches a normalized query against the corpus using cosine similarity.
        Returns:
            (best_faq_dict, similarity_score) if score >= threshold
            (None, similarity_score) if score < threshold
        """
        clean_query = normalize_text(query)
        if not clean_query:
            return None, 0.0

        # Auto-correct typographical errors before vectorization
        clean_query = self.auto_correct_query(clean_query)

        query_vec = self.vectorizer.transform([clean_query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        if len(similarities) == 0:
            return None, 0.0

        best_idx = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])

        # Specific unsupported query guard (exact package prediction, personal confidential info)
        unsupported_intent_patterns = [
            r"exact placement package offered to me",
            r"how much salary will i get next year",
            r"what will be my personal rank",
            r"book (a|the) (train|flight|movie|bus) ticket",
            r"what is the weather today",
            r"who is the president"
        ]
        for pattern in unsupported_intent_patterns:
            if re.search(pattern, clean_query):
                return None, 0.0

        if best_score >= threshold:
            return self.faq_list[best_idx], best_score
        else:
            return None, best_score


def generate_rag_answer(user_query: str, faq_match: Dict[str, Any], api_key: str) -> str:
    """
    Optional Generative AI layer:
    Prompts Gemini to synthesize a polite, natural student answer strictly
    grounded in the verified FAQ context.
    """
    if not GENAI_AVAILABLE or not api_key:
        return faq_match["answer"]

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")

        prompt = f"""
You are the official AI FAQ Assistant for the College MCA Department.
Your task is to answer the student's question accurately, politely, and STRICTLY using only the verified Departmental FAQ context provided below.

RULES:
1. Do NOT assume, invent, extrapolate, or hallucinate policies, dates, marks, fees, or figures.
2. If the context does not fully answer the student query, politely state that verified information is not available and advise them to consult the department office.
3. Keep the response concise, clear, and student-friendly.

VERIFIED DEPARTMENTAL FAQ CONTEXT:
Category: {faq_match.get('category')}
Question: {faq_match.get('question')}
Verified Answer: {faq_match.get('answer')}

STUDENT'S QUERY:
{user_query}

ANSWER:
"""
        response = model.generate_content(prompt)
        if response and response.text:
            return response.text.strip()
        return faq_match["answer"]
    except Exception as e:
        # Graceful fallback to verified JSON answer if API fails
        return f"{faq_match['answer']}\n\n*(Note: Generative AI layer fallback triggered: {str(e)})*"


def process_user_query(
    query: str,
    retriever: FAQRetriever,
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    gemini_api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    End-to-end query processing function for UI and automated testing.
    """
    matched_faq, score = retriever.retrieve(query, threshold=threshold)

    if matched_faq is None:
        return {
            "answer": FALLBACK_RESPONSE,
            "status": "FALLBACK",
            "score": score,
            "matched_faq": None,
            "category": "Unknown",
            "mode": "Rule/Fallback"
        }

    # If Generative AI key is available, use RAG synthesis; else direct verified answer
    if gemini_api_key and GENAI_AVAILABLE:
        answer_text = generate_rag_answer(query, matched_faq, gemini_api_key)
        mode = "Generative AI (Gemini RAG)"
    else:
        answer_text = matched_faq["answer"]
        mode = "Local TF-IDF Matcher"

    return {
        "answer": answer_text,
        "status": "SUCCESS",
        "score": score,
        "matched_faq": matched_faq,
        "category": matched_faq.get("category", "General"),
        "mode": mode
    }


# =============================================================================
# 3. STREAMLIT WEB INTERFACE
# =============================================================================
def init_page_config():
    st.set_page_config(
        page_title="MCA Dept AI FAQ Chatbot",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom styling for academic practical examination presentation
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.1rem;
            font-weight: 700;
            color: #1E3A8A;
            margin-bottom: 0.2rem;
        }
        .sub-header {
            font-size: 1.05rem;
            color: #4B5563;
            margin-bottom: 1.5rem;
        }
        .badge-local {
            background-color: #E0F2FE;
            color: #0369A1;
            padding: 4px 10px;
            border-radius: 6px;
            font-weight: 600;
            font-size: 0.85rem;
            display: inline-block;
        }
        .badge-genai {
            background-color: #FEF3C7;
            color: #B45309;
            padding: 4px 10px;
            border-radius: 6px;
            font-weight: 600;
            font-size: 0.85rem;
            display: inline-block;
        }
        .faq-card {
            border-left: 4px solid #2563EB;
            padding: 10px 14px;
            background-color: #F8FAFC;
            border-radius: 4px;
            margin-bottom: 8px;
        }
        </style>
    """, unsafe_allow_html=True)


def main():
    init_page_config()

    # Load Knowledge Base
    try:
        faq_data = load_faq_database()
        retriever = FAQRetriever(faq_data)
    except Exception as e:
        st.error(f"❌ Failed to load FAQ Knowledge Base: {e}")
        st.stop()

    # Sidebar: Configurations & Project Information
    with st.sidebar:
        st.header("⚙️ Project Controls")
        st.markdown("**MCA GenAI Lab Practical Exam**")
        st.caption("Department of Computer Applications")
        st.markdown("---")

        # Similarity threshold slider
        similarity_threshold = st.slider(
            "Cosine Similarity Threshold",
            min_value=0.10,
            max_value=0.60,
            value=DEFAULT_SIMILARITY_THRESHOLD,
            step=0.02,
            help="Queries with similarity below this threshold trigger the polite fallback response."
        )

        st.markdown("---")
        st.subheader("🔑 Generative AI Configuration")
        st.caption("Optional: Enter a Gemini API Key for RAG synthesis. If left empty, local TF-IDF matching is used (Zero Cost, 100% Offline).")
        api_key_input = st.text_input(
            "Gemini API Key (Optional)",
            type="password",
            value=os.getenv("GEMINI_API_KEY", ""),
            help="Free API key from Google AI Studio (aistudio.google.com). Optional."
        )

        st.markdown("---")
        st.subheader("📊 Knowledge Base Info")
        st.write(f"• **Total FAQs Loaded:** {len(faq_data)}")
        categories = sorted(list(set(item["category"] for item in faq_data)))
        st.write(f"• **Categories ({len(categories)}):**")
        st.caption(", ".join(categories))

        with st.expander("🔍 View Knowledge Base"):
            selected_cat = st.selectbox("Filter category", ["All"] + categories)
            for item in faq_data:
                if selected_cat == "All" or item["category"] == selected_cat:
                    st.markdown(f"**[{item['category']}]** {item['question']}")
                    st.caption(item['answer'])
                    st.markdown("---")

        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    # Main Chat Area
    st.markdown('<div class="main-header">🎓 AI-Powered FAQ Chatbot</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Department of Computer Applications (MCA) — Student Enquiry Assistant</div>', unsafe_allow_html=True)

    # Active engine badge
    if api_key_input and GENAI_AVAILABLE:
        st.markdown('<span class="badge-genai">✨ Mode: Hybrid RAG (TF-IDF Retrieval + Gemini Generative AI)</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge-local">⚡ Mode: Local TF-IDF Vectorizer (Zero API Cost / Deterministic Grounding)</span>', unsafe_allow_html=True)

    st.write("")

    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Hello! 👋 I am your **MCA Department AI Assistant**.\n\n"
                    "I can answer queries on **Admissions, Courses, Examinations, Attendance, Fees, Placements, and Facilities** strictly from our official FAQ knowledge base.\n\n"
                    "How can I assist you today?"
                ),
                "metadata": None
            }
        ]

    # Display Suggested Questions for Quick Testing during Viva
    st.markdown("##### 💡 Suggested Questions (Click to test):")
    cols = st.columns(3)
    sample_queries = [
        "What is the eligibility criteria for MCA admission?",
        "What are the elective subjects offered in MCA?",
        "What is the minimum attendance required for exams?",
        "How does the placement cell assist MCA students?",
        "What are the library timings and digital facilities?",
        "What will be the exact placement package offered to me next year?"
    ]

    selected_prompt = None
    for i, sq in enumerate(sample_queries):
        col = cols[i % 3]
        if col.button(sq, key=f"btn_sq_{i}", use_container_width=True):
            selected_prompt = sq

    st.markdown("---")

    # Render Chat History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("metadata"):
                meta = message["metadata"]
                if meta["status"] == "SUCCESS":
                    st.caption(
                        f"🎯 *Matched Category:* `{meta['category']}` | "
                        f"*Similarity:* `{meta['score']:.3f}` | "
                        f"*Engine:* `{meta['mode']}`"
                    )
                else:
                    st.caption(f"⚠️ *Query Status:* `Unsupported/Fallback` | *Score:* `{meta['score']:.3f}`")

    # Handle Input (from chat_input or suggested button click)
    user_input = st.chat_input("Type your question about admissions, courses, exams, placements...")
    prompt_to_process = selected_prompt if selected_prompt else user_input

    if prompt_to_process:
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt_to_process, "metadata": None})
        with st.chat_message("user"):
            st.markdown(prompt_to_process)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Searching knowledge base..."):
                result = process_user_query(
                    query=prompt_to_process,
                    retriever=retriever,
                    threshold=similarity_threshold,
                    gemini_api_key=api_key_input
                )

                st.markdown(result["answer"])
                if result["status"] == "SUCCESS":
                    st.caption(
                        f"🎯 *Matched Category:* `{result['category']}` | "
                        f"*Similarity:* `{result['score']:.3f}` | "
                        f"*Engine:* `{result['mode']}`"
                    )
                else:
                    st.caption(f"⚠️ *Query Status:* `Unsupported/Fallback` | *Score:* `{result['score']:.3f}`")

        # Save assistant message
        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "metadata": result
        })


if __name__ == "__main__":
    main()
