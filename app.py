import os
import base64
import random

# Make Python's SSL stack trust the OS certificate store (Windows / macOS keychain / Linux).
# This is required behind corporate proxies that perform SSL inspection — without it,
# huggingface_hub's httpx client fails to download the embedding model with
# "self-signed certificate in certificate chain" / "client has been closed".
try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass

import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

load_dotenv()


def _get_secret(key: str, default: str = "") -> str:
    """Read a secret from st.secrets (Streamlit Cloud) or os.environ (local .env)."""
    try:
        return st.secrets[key]
    except Exception:
        return os.getenv(key, default)


# Groq model is configurable so we can swap when a model is deprecated
# without a code push. Override via the GROQ_MODEL env var (or Streamlit secret).
GROQ_MODEL = _get_secret("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Tableau Genie",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =====================================================
# CUSTOM CSS — TABLEAU BLUE BRANDED
# =====================================================

st.markdown(
    """
<style>

/* ---- Tableau brand palette ----
   --tab-navy    : #0F4C81   (deep brand navy)
   --tab-blue    : #1F77B4   (classic Tableau blue)
   --tab-sky     : #5BB6E8   (light accent)
   --tab-orange  : #E97627   (accent / CTA)
   --tab-ink     : #0F1B2D   (body text)
   --tab-cloud   : #F5F8FB   (page background)
*/

#MainMenu        { visibility: hidden; }
footer           { visibility: hidden; }
header           { visibility: hidden; }

/* ---------- Page background (dark Tableau-navy) ---------- */
.stApp {
    background:
        radial-gradient(circle at 0% 0%,   rgba(31,119,180,0.20), transparent 40%),
        radial-gradient(circle at 100% 0%, rgba(91,182,232,0.14), transparent 45%),
        radial-gradient(circle at 50% 100%,rgba(233,118,39,0.08), transparent 50%),
        linear-gradient(180deg, #06101F 0%, #0A1A33 50%, #06101F 100%);
    color: #D4DEEA;
}

/* Constrain main column for readability */
.block-container {
    max-width: 1100px;
    padding-top: 1.5rem;
    padding-bottom: 6rem;
}

/* ---------- Sidebar (slightly lighter than main bg for separation) ---------- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0F2E4F 0%, #0A2038 100%);
    border-right: 1px solid rgba(91,182,232,0.16);
}
[data-testid="stSidebar"] * {
    color: #EAF2F9 !important;
}
[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.12);
}
.sidebar-title {
    font-size: 1.4rem;
    font-weight: 800;
    letter-spacing: 0.3px;
    margin: 8px 0 2px 0;
    color: #FFFFFF !important;
}
.sidebar-tag {
    font-size: 0.86rem;
    color: #B9D4E8 !important;
    margin-bottom: 10px;
}
.sidebar-section {
    text-transform: uppercase;
    font-size: 0.72rem;
    letter-spacing: 1.4px;
    color: #8FB6D4 !important;
    margin: 6px 0 8px 0;
}
.tech-badges {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-top: 4px;
}
.tech-badge {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 999px;
    font-size: 0.74rem;
    font-weight: 600;
    background: rgba(255,255,255,0.10);
    border: 1px solid rgba(255,255,255,0.18);
    color: #EAF2F9 !important;
}

/* ----- DEFAULT button style (used for welcome cards in main area) -----
   We style ALL Streamlit buttons here, then override the sidebar variant below
   (sidebar wins on specificity).  We use !important throughout because Streamlit
   ships its own button CSS with high specificity that would otherwise win. */
.stButton > button,
[data-testid="stBaseButton-secondary"] {
    width: 100%;
    background: linear-gradient(180deg, rgba(19,40,74,0.88) 0%, rgba(13,30,57,0.88) 100%) !important;
    color: #EAF2F9 !important;
    border: 1px solid rgba(91,182,232,0.22) !important;
    border-radius: 16px !important;
    padding: 20px 20px !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    text-align: left !important;
    line-height: 1.45 !important;
    box-shadow: 0 4px 18px rgba(0,0,0,0.30), inset 0 1px 0 rgba(255,255,255,0.05) !important;
    backdrop-filter: blur(8px);
    transition: all 0.2s ease !important;
    white-space: normal !important;
}
.stButton > button:hover,
[data-testid="stBaseButton-secondary"]:hover {
    border-color: #5BB6E8 !important;
    background: linear-gradient(180deg, rgba(31,119,180,0.32) 0%, rgba(15,76,129,0.32) 100%) !important;
    box-shadow:
        0 8px 28px rgba(31,119,180,0.40),
        0 0 0 1px rgba(91,182,232,0.45) !important;
    transform: translateY(-2px);
    color: #FFFFFF !important;
}
.stButton > button:focus:not(:active),
[data-testid="stBaseButton-secondary"]:focus:not(:active) {
    border-color: #5BB6E8 !important;
    color: #FFFFFF !important;
    box-shadow: 0 0 0 3px rgba(91,182,232,0.25) !important;
}
/* Force inner paragraph and span elements inside every button to use the button color */
.stButton > button p,
.stButton > button span,
[data-testid="stBaseButton-secondary"] p,
[data-testid="stBaseButton-secondary"] span {
    color: inherit !important;
    margin: 0;
}

/* ----- Sidebar button override (smaller, more compact) ----- */
[data-testid="stSidebar"] .stButton > button,
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.16) !important;
    border-radius: 12px !important;
    padding: 10px 12px !important;
    font-size: 0.86rem !important;
    font-weight: 500 !important;
    line-height: 1.35 !important;
    box-shadow: none !important;
    backdrop-filter: none;
    color: #EAF2F9 !important;
}
[data-testid="stSidebar"] .stButton > button:hover,
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]:hover {
    background: rgba(91,182,232,0.20) !important;
    border-color: #5BB6E8 !important;
    box-shadow: none !important;
    transform: translateY(-1px);
    color: #FFFFFF !important;
}

/* ---------- Header / Title ---------- */
.header-wrap {
    text-align: center;
    margin: 8px 0 30px 0;
}
.header-logo {
    display: block;
    margin: 0 auto 14px auto;
    height: 64px;
    width: auto;
    filter: drop-shadow(0 0 16px rgba(91,182,232,0.40));
}
.main-title {
    font-size: 3.6rem;
    font-weight: 900;
    letter-spacing: -0.5px;
    line-height: 1.05;
    margin: 0;
    color: #FFFFFF;
    text-shadow: 0 0 30px rgba(91,182,232,0.18);
}
.main-title .accent {
    background: linear-gradient(90deg, #5BB6E8 0%, #1F77B4 50%, #E97627 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.subtitle {
    font-size: 1.05rem;
    color: #8FB6D4;
    margin-top: 8px;
    margin-bottom: 0;
    font-weight: 500;
}
.title-divider {
    width: 80px;
    height: 4px;
    background: linear-gradient(90deg, #1F77B4, #E97627);
    border-radius: 999px;
    margin: 14px auto 0 auto;
}

/* ---------- Welcome / example cards ---------- */
.welcome-heading {
    text-align: center;
    color: #5BB6E8;
    font-weight: 700;
    font-size: 1.05rem;
    margin: 24px 0 10px 0;
    letter-spacing: 0.3px;
}
.welcome-sub {
    text-align: center;
    color: #8FB6D4;
    font-size: 0.92rem;
    margin-bottom: 22px;
}

/* (welcome-card / main-area button styles handled by the global .stButton rule above) */

/* ---------- Chat messages (dark glassy cards) ---------- */
[data-testid="stChatMessage"] {
    background: linear-gradient(180deg, rgba(19,40,74,0.55) 0%, rgba(13,30,57,0.55) 100%);
    border: 1px solid rgba(91,182,232,0.18);
    border-radius: 18px;
    padding: 16px 18px;
    margin-bottom: 14px;
    backdrop-filter: blur(10px);
    box-shadow: 0 4px 22px rgba(0,0,0,0.28);
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background: linear-gradient(180deg, rgba(31,119,180,0.20) 0%, rgba(15,76,129,0.18) 100%);
    border-color: rgba(91,182,232,0.32);
}
[data-testid="stChatMessageContent"] p,
[data-testid="stChatMessageContent"] li,
[data-testid="stChatMessageContent"] span {
    color: #D4DEEA !important;
    font-size: 0.97rem;
    line-height: 1.6;
}
[data-testid="stChatMessageContent"] h1,
[data-testid="stChatMessageContent"] h2,
[data-testid="stChatMessageContent"] h3,
[data-testid="stChatMessageContent"] h4 {
    color: #5BB6E8 !important;
}
[data-testid="stChatMessageContent"] strong {
    color: #FFFFFF !important;
}
[data-testid="stChatMessageContent"] table {
    background: rgba(8,21,42,0.45);
    border-radius: 8px;
    overflow: hidden;
    margin: 10px 0;
}
[data-testid="stChatMessageContent"] th {
    background: rgba(31,119,180,0.28) !important;
    color: #FFFFFF !important;
    border: 1px solid rgba(91,182,232,0.18) !important;
}
[data-testid="stChatMessageContent"] td {
    color: #D4DEEA !important;
    border: 1px solid rgba(91,182,232,0.10) !important;
}

/* Inline code in prose (e.g. `column_name`).
   The :not(pre code) guard is intentional and important: Streamlit
   nests code blocks one level deeper than plain HTML, so a parent-only
   check leaks this pill styling onto every Prism token inside SQL
   code blocks. */
[data-testid="stChatMessageContent"] code:not(pre code) {
    background: rgba(31,119,180,0.18);
    color: #5BB6E8 !important;
    padding: 2px 6px;
    border-radius: 6px;
    font-size: 0.88em;
    font-family: 'Cascadia Code', 'Fira Code', 'JetBrains Mono', Consolas, monospace;
    border: 1px solid rgba(91,182,232,0.22);
}

/* ---- Code-block container (VS Code-style dark editor card) ---- */
[data-testid="stChatMessageContent"] pre {
    background: #1E1E1E !important;
    border: 1px solid rgba(91,182,232,0.22);
    border-radius: 12px;
    padding: 38px 18px 16px 18px !important;
    box-shadow: 0 6px 22px rgba(15,76,129,0.18);
    position: relative;
    overflow-x: auto;
    margin: 14px 0 !important;
}
[data-testid="stChatMessageContent"] pre code {
    background: transparent !important;
    font-family: 'Cascadia Code', 'Fira Code', 'JetBrains Mono', Consolas, 'Courier New', monospace !important;
    font-size: 0.86rem !important;
    line-height: 1.6;
    color: #D4D4D4;
}

/* "Window chrome" dots in the top-left of every code block */
[data-testid="stChatMessageContent"] pre::before {
    content: '';
    position: absolute;
    top: 12px;
    left: 14px;
    width: 12px;
    height: 12px;
    border-radius: 50%;
    background: #FF5F56;
    box-shadow:
        18px 0 0 #FFBD2E,
        36px 0 0 #27C93F;
    opacity: 0.9;
}

/* Language label pill (top-right) */
[data-testid="stChatMessageContent"] pre:has(code[class*="language-sql"])::after  { content: 'SQL';        background: #1F77B4; }
[data-testid="stChatMessageContent"] pre:has(code[class*="language-python"])::after { content: 'PYTHON'; background: #3776AB; }
[data-testid="stChatMessageContent"] pre:has(code[class*="language-json"])::after   { content: 'JSON';   background: #6A737D; }
[data-testid="stChatMessageContent"] pre:has(code[class*="language-bash"])::after,
[data-testid="stChatMessageContent"] pre:has(code[class*="language-sh"])::after,
[data-testid="stChatMessageContent"] pre:has(code[class*="language-shell"])::after  { content: 'SHELL';  background: #4D4D4D; }
[data-testid="stChatMessageContent"] pre:has(code[class*="language-"])::after {
    position: absolute;
    top: 10px;
    right: 12px;
    color: #FFFFFF;
    font-size: 0.65rem;
    font-weight: 700;
    padding: 2px 10px;
    border-radius: 999px;
    letter-spacing: 1.2px;
    font-family: -apple-system, system-ui, sans-serif;
}

/* ===========================================================
   Syntax-highlighter theme — Tableau-tinted "VS Code Dark+"
   Covers Prism, react-syntax-highlighter and highlight.js classes
   so it works whichever renderer Streamlit uses.
   =========================================================== */

/* Comments */
[data-testid="stChatMessageContent"] pre code .token.comment,
[data-testid="stChatMessageContent"] pre code .token.prolog,
[data-testid="stChatMessageContent"] pre code .token.doctype,
[data-testid="stChatMessageContent"] pre code .token.cdata,
[data-testid="stChatMessageContent"] pre code .hljs-comment,
[data-testid="stChatMessageContent"] pre code .hljs-quote {
    color: #6A9955 !important;
    font-style: italic;
}

/* Keywords (SELECT, FROM, WHERE, JOIN, etc.) */
[data-testid="stChatMessageContent"] pre code .token.keyword,
[data-testid="stChatMessageContent"] pre code .token.atrule,
[data-testid="stChatMessageContent"] pre code .token.attr-value,
[data-testid="stChatMessageContent"] pre code .hljs-keyword,
[data-testid="stChatMessageContent"] pre code .hljs-selector-tag,
[data-testid="stChatMessageContent"] pre code .hljs-section {
    color: #5BB6E8 !important;
    font-weight: 600;
}

/* Strings */
[data-testid="stChatMessageContent"] pre code .token.string,
[data-testid="stChatMessageContent"] pre code .token.char,
[data-testid="stChatMessageContent"] pre code .token.attr-name,
[data-testid="stChatMessageContent"] pre code .token.selector,
[data-testid="stChatMessageContent"] pre code .token.inserted,
[data-testid="stChatMessageContent"] pre code .hljs-string,
[data-testid="stChatMessageContent"] pre code .hljs-attr,
[data-testid="stChatMessageContent"] pre code .hljs-symbol {
    color: #CE9178 !important;
}

/* Numbers, booleans, NULL */
[data-testid="stChatMessageContent"] pre code .token.number,
[data-testid="stChatMessageContent"] pre code .token.boolean,
[data-testid="stChatMessageContent"] pre code .hljs-number,
[data-testid="stChatMessageContent"] pre code .hljs-literal {
    color: #B5CEA8 !important;
}

/* Functions (COUNT, SUM, AVG, ...) */
[data-testid="stChatMessageContent"] pre code .token.function,
[data-testid="stChatMessageContent"] pre code .token.class-name,
[data-testid="stChatMessageContent"] pre code .hljs-title,
[data-testid="stChatMessageContent"] pre code .hljs-built_in {
    color: #DCDCAA !important;
}

/* Identifiers / column / table names */
[data-testid="stChatMessageContent"] pre code .token.property,
[data-testid="stChatMessageContent"] pre code .token.tag,
[data-testid="stChatMessageContent"] pre code .token.constant,
[data-testid="stChatMessageContent"] pre code .token.variable,
[data-testid="stChatMessageContent"] pre code .hljs-variable,
[data-testid="stChatMessageContent"] pre code .hljs-name,
[data-testid="stChatMessageContent"] pre code .hljs-attribute {
    color: #9CDCFE !important;
}

/* Operators / punctuation */
[data-testid="stChatMessageContent"] pre code .token.operator,
[data-testid="stChatMessageContent"] pre code .token.punctuation,
[data-testid="stChatMessageContent"] pre code .token.entity,
[data-testid="stChatMessageContent"] pre code .hljs-operator,
[data-testid="stChatMessageContent"] pre code .hljs-punctuation {
    color: #D4D4D4 !important;
}

/* Builtins / types */
[data-testid="stChatMessageContent"] pre code .token.builtin,
[data-testid="stChatMessageContent"] pre code .hljs-type {
    color: #4EC9B0 !important;
}

/* Errors / important */
[data-testid="stChatMessageContent"] pre code .token.regex,
[data-testid="stChatMessageContent"] pre code .token.important,
[data-testid="stChatMessageContent"] pre code .token.deleted {
    color: #D16969 !important;
}

/* ---------- Sources expander (dark) ---------- */
[data-testid="stChatMessage"] [data-testid="stExpander"] {
    background: rgba(8,21,42,0.55);
    border: 1px solid rgba(91,182,232,0.20);
    border-radius: 12px;
    margin-top: 10px;
}
[data-testid="stChatMessage"] [data-testid="stExpander"] summary {
    color: #5BB6E8 !important;
    font-weight: 600;
    font-size: 0.88rem;
}
.source-chunk {
    background: rgba(13,30,57,0.55);
    border-left: 3px solid #5BB6E8;
    border-radius: 8px;
    padding: 10px 14px;
    margin: 6px 0 12px 0;
    font-size: 0.86rem;
    color: #C4D2E2;
    line-height: 1.5;
}
.source-label {
    display: inline-block;
    background: #1F77B4;
    color: #FFFFFF !important;
    padding: 2px 10px;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.5px;
    margin-bottom: 6px;
}

/* ---------- Chat input (dark glassy, single rounded card) ---------- */
[data-testid="stChatInput"] {
    background: transparent;
}
/* The visible card is the outer wrapper; carries the only border + radius */
[data-testid="stChatInput"] > div {
    background: linear-gradient(180deg, rgba(13,30,57,0.85) 0%, rgba(8,21,42,0.85) 100%) !important;
    border: 1.5px solid rgba(91,182,232,0.26) !important;
    border-radius: 16px !important;
    backdrop-filter: blur(10px);
    box-shadow: 0 4px 18px rgba(0,0,0,0.30) !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
    overflow: hidden;
}
[data-testid="stChatInput"] > div:focus-within {
    border-color: #5BB6E8 !important;
    box-shadow: 0 0 0 3px rgba(91,182,232,0.22) !important;
}
/* The textarea inside is transparent + borderless so we don't get double-borders */
[data-testid="stChatInput"] textarea {
    background: transparent !important;
    border: none !important;
    border-radius: 0 !important;
    color: #EAF2F9 !important;
    font-size: 0.97rem !important;
    box-shadow: none !important;
    padding: 14px 18px !important;
    line-height: 1.5 !important;
    box-sizing: border-box !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: #6F8AAB !important;
}
[data-testid="stChatInput"] textarea:focus,
[data-testid="stChatInput"] textarea:focus-visible {
    outline: none !important;
    border: none !important;
    box-shadow: none !important;
}
[data-testid="stChatInput"] button {
    color: #5BB6E8 !important;
}

/* Spinner color */
[data-testid="stSpinner"] > div > div {
    border-top-color: #5BB6E8 !important;
}
[data-testid="stSpinner"] {
    color: #8FB6D4 !important;
}

/* General body text colour reset for main area
   (deliberately scoped to direct children so it does NOT bleed into code blocks) */
.block-container > div p,
.block-container > div label {
    color: #D4DEEA;
}

/* Belt-and-braces: every plain span inside a code block uses the editor default colour
   so unstyled identifiers (column names, table names) stay readable on the dark editor
   background. Tokenised spans (.token.keyword etc.) override this above. */
[data-testid="stChatMessageContent"] pre code,
[data-testid="stChatMessageContent"] pre code span {
    color: #D4D4D4 !important;
}
[data-testid="stChatMessageContent"] pre code .token {
    /* let token-specific colours win */
    color: inherit;
}

</style>
""",
    unsafe_allow_html=True,
)

# =====================================================
# CONFIG
# =====================================================

COLLECTION_NAME = "tableau_docs"

LOGO_PATH = "Tableau_logo.png"

# =====================================================
# LOAD MODELS
# =====================================================

@st.cache_resource(show_spinner=False)
def load_embedding_model():
    return SentenceTransformer("BAAI/bge-small-en-v1.5")


@st.cache_resource(show_spinner=False)
def load_qdrant():
    return QdrantClient(path="./qdrant_data")


@st.cache_resource(show_spinner=False)
def load_groq():
    api_key = _get_secret("GROQ_API_KEY")
    if not api_key:
        st.error(
            "⚠️ **GROQ_API_KEY is not set.** "
            "Go to **Manage app → Settings → Secrets** and add:\n\n"
            "```toml\nGROQ_API_KEY = \"gsk_your_key_here\"\n```"
        )
        st.stop()
    return Groq(api_key=api_key)


# =====================================================
# RETRIEVAL
# =====================================================

def retrieve(query, top_k=5):
    embedding_model = load_embedding_model()
    client = load_qdrant()

    query_vector = embedding_model.encode(query).tolist()

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
    )

    return results.points


# =====================================================
# ANSWER QUESTION
# =====================================================

def _build_prompt(question, context):
    return f"""
You are Tableau Genie.

Use the supplied context as the primary source.

For every answer provide:

1. Explanation
2. Sample Data
3. Tableau Calculation
4. Equivalent SQL Query
5. Expected Output
6. When To Use

Format all SQL queries inside ```sql code blocks.

If the documentation does not contain an example,
create a realistic business example using fictional data.

CONTEXT:
{context}

QUESTION:
{question}
"""


def stream_answer(question):
    """Return (token_generator, docs) for live-streaming a Groq answer to the UI."""
    groq_client = load_groq()
    docs = retrieve(question)
    context = "\n\n".join(doc.payload["content"] for doc in docs)
    prompt = _build_prompt(question, context)

    stream = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        stream=True,
    )

    def token_iter():
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    return token_iter(), docs


def answer_question(question):
    """Non-streaming variant kept for backwards compatibility."""
    groq_client = load_groq()
    docs = retrieve(question)
    context = "\n\n".join(doc.payload["content"] for doc in docs)
    prompt = _build_prompt(question, context)

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    return response.choices[0].message.content, docs


# =====================================================
# HELPERS
# =====================================================

SAMPLE_QUESTIONS = [
    "How do I create a calculated field?",
    "What is a FIXED LOD expression?",
    "How do parameters work in Tableau?",
    "What is Tableau Web Authoring?",
    "How do I create a dual-axis chart?",
    "Difference between LOD INCLUDE and EXCLUDE?",
]

SPINNER_LINES = [
    "Polishing the lamp…",
    "Consulting the Tableau scrolls…",
    "Summoning the data spirits…",
    "Sprinkling a bit of Tableau magic…",
    "Channelling years of Tableau wisdom…",
    "The Genie is pondering deeply…",
    "Rummaging through the dashboard archives…",
    "Asking the LOD oracle…",
    "Whispering with the data spirits…",
    "Brewing a fresh batch of insights…",
]

EXAMPLE_CARDS = [
    ("📐  How do I create a calculated field with conditional logic?"),
    ("🎯  What is a FIXED LOD expression and when should I use it?"),
    ("⚙️  How do parameters work in Tableau and how do I use them?"),
    ("🌐  What is Tableau Web Authoring vs Tableau Desktop?"),
]


@st.cache_data
def _logo_b64():
    """Return the Tableau logo as a base64 data URI fragment (cached)."""
    if not os.path.exists(LOGO_PATH):
        return ""
    with open(LOGO_PATH, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def render_sources(sources):
    """Render a list of source chunks (strings) inside an expander."""
    if not sources:
        return
    with st.expander(f"📚  Sources from Tableau documentation ({len(sources)})"):
        for i, chunk in enumerate(sources, 1):
            st.markdown(
                f"<span class='source-label'>SOURCE {i}</span>",
                unsafe_allow_html=True,
            )
            preview = chunk if len(chunk) <= 700 else chunk[:700].rstrip() + "…"
            st.markdown(
                f"<div class='source-chunk'>{preview}</div>",
                unsafe_allow_html=True,
            )


# =====================================================
# SESSION STATE
# =====================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=160)

    st.markdown(
        "<div class='sidebar-title'>Tableau Genie</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='sidebar-tag'>RAG-powered assistant for Tableau Desktop &amp; Web Authoring documentation.</div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    st.markdown(
        "<div class='sidebar-section'>Try asking</div>",
        unsafe_allow_html=True,
    )
    for i, q in enumerate(SAMPLE_QUESTIONS):
        if st.button(q, key=f"side_q_{i}"):
            st.session_state.pending_question = q
            st.rerun()

    st.markdown("---")

    st.markdown(
        "<div class='sidebar-section'>Tech stack</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        """
<div class='tech-badges'>
  <span class='tech-badge'>Streamlit</span>
  <span class='tech-badge'>Qdrant</span>
  <span class='tech-badge'>BGE Embeddings</span>
  <span class='tech-badge'>Groq · Llama 3.3 70B</span>
  <span class='tech-badge'>PyMuPDF</span>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    if st.button("🗑  Clear conversation", key="clear_chat"):
        st.session_state.messages = []
        st.session_state.pending_question = None
        st.rerun()

    st.markdown(
        "<div style='margin-top:18px; font-size:0.74rem; color:#8FB6D4;'>"
        "Built with RAG · Answers grounded in Tableau docs"
        "</div>",
        unsafe_allow_html=True,
    )


# =====================================================
# HEADER
# =====================================================

_logo_data = _logo_b64()
_logo_html = (
    f"<img src='data:image/png;base64,{_logo_data}' class='header-logo' alt='Tableau' />"
    if _logo_data
    else ""
)

st.markdown(
    f"""
<div class='header-wrap'>
  {_logo_html}
  <h1 class='main-title'>Tableau <span class='accent'>Genie</span></h1>
  <p class='subtitle'>Your AI-powered assistant for Tableau Desktop &amp; Web Authoring documentation</p>
  <div class='title-divider'></div>
</div>
""",
    unsafe_allow_html=True,
)


# =====================================================
# WELCOME / EXAMPLE CARDS (only when no chat yet)
# =====================================================

if not st.session_state.messages and not st.session_state.pending_question:
    st.markdown(
        "<div class='welcome-heading'>Get started with a quick question</div>"
        "<div class='welcome-sub'>Click any prompt below or type your own in the chat box.</div>",
        unsafe_allow_html=True,
    )

    cols = st.columns(2)
    for i, label in enumerate(EXAMPLE_CARDS):
        with cols[i % 2]:
            if st.button(label, key=f"ex_{i}"):
                # Strip leading icon + spaces for the actual question
                clean = label.split("  ", 1)[-1] if "  " in label else label
                st.session_state.pending_question = clean
                st.rerun()


# =====================================================
# CHAT HISTORY
# =====================================================

for msg in st.session_state.messages:
    avatar = "🧑‍💼" if msg["role"] == "user" else "✨"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_sources(msg.get("sources", []))


# =====================================================
# CHAT INPUT
# =====================================================

typed = st.chat_input("Ask Tableau Genie about LOD, calculated fields, parameters, dashboards…")

# A click on a sample / example button also feeds in a question
question = typed or st.session_state.pending_question
if st.session_state.pending_question:
    st.session_state.pending_question = None

if question:
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user", avatar="🧑‍💼"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="✨"):
        with st.spinner(random.choice(SPINNER_LINES)):
            token_stream, docs = stream_answer(question)

        # Tokens appear live as Groq generates them
        answer = st.write_stream(token_stream)

        sources = [doc.payload.get("content", "") for doc in docs]
        render_sources(sources)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
        }
    )
