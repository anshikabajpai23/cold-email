import os
import shutil
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from generator import extract_relevant_facts, generate, list_templates, rank_relevance, render_prompt
from rag import build_index, query_background

DOCS_DIR = Path(__file__).parent / "docs"
CACHE_DIR = Path(__file__).parent / "cache"

# Clear vector index cache on every app start so stale embeddings never persist
if CACHE_DIR.exists():
    shutil.rmtree(CACHE_DIR)

st.set_page_config(page_title="Cover Letter Generator", page_icon="✉️", layout="centered")

st.title("Cover Letter Generator")
st.caption("Personalized outreach powered by your background docs.")

# ── Sidebar ────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Background Docs")

    uploaded = st.file_uploader(
        "Upload resume / bio / project writeups",
        type=["txt", "pdf", "md"],
        accept_multiple_files=True,
    )

    if uploaded:
        DOCS_DIR.mkdir(exist_ok=True)
        for f in uploaded:
            dest = DOCS_DIR / f.name
            dest.write_bytes(f.read())
        st.success(f"{len(uploaded)} file(s) saved to docs/")

    existing = list(DOCS_DIR.glob("*")) if DOCS_DIR.exists() else []
    if existing:
        st.markdown("**Docs loaded:**")
        for p in existing:
            st.markdown(f"- {p.name}")

    if st.button("Build / Rebuild Index", type="primary"):
        with st.spinner("Indexing docs (first run downloads ~90MB model)..."):
            try:
                build_index()
                st.success("Index built!")
            except FileNotFoundError as e:
                st.error(str(e))

    st.divider()
    st.header("Settings")
    model = st.selectbox("Ollama model", ["qwen2.5:14b", "llama3.1", "llama3.2"], index=0)
    st.caption("Make sure Ollama is running: `ollama serve`")

# ── Main form ──────────────────────────────────────────────────────────────

available_templates = list_templates()
template_labels = {
    "cover_letter": "Cover Letter",
    "linkedin_message": "LinkedIn Message",
}

template = st.segmented_control(
    "Output type",
    options=available_templates,
    format_func=lambda t: template_labels.get(t, t.replace("_", " ").title()),
    default=available_templates[0] if available_templates else None,
)

st.divider()

col1, col2 = st.columns(2)
with col1:
    company = st.text_input("Company *", placeholder="Stripe")
with col2:
    role = st.text_input("Role *", placeholder="Product Manager")

col3, col4, col5 = st.columns(3)
with col3:
    person_name = st.text_input("Contact name", placeholder="Sarah")
with col4:
    person_title = st.text_input("Their title", placeholder="Recruiter")
with col5:
    contact_email = st.text_input("Their email", placeholder="sarah@stripe.com")

jd_text = st.text_area("Job Description (optional)", height=160, placeholder="Paste the JD here...")

st.divider()

col_gen, col_preview = st.columns([3, 1])
with col_gen:
    generate_btn = st.button("Generate", type="primary", use_container_width=True)
with col_preview:
    preview_btn = st.button("Preview Prompt", use_container_width=True)

if preview_btn or generate_btn:
    if not company or not role:
        st.error("Company and Role are required.")
        st.stop()
    if not DOCS_DIR.exists() or not any(DOCS_DIR.iterdir()):
        st.error("No docs found. Upload your resume/background docs in the sidebar first.")
        st.stop()

    with st.spinner("Step 1/3 — Retrieving background from docs..."):
        rag_query = f"{role} at {company}. {jd_text[:500]}"
        try:
            raw_background = query_background(rag_query)
        except Exception as e:
            st.error(f"Error reading docs: {e}")
            st.stop()

    with st.expander("Debug: Raw RAG context"):
        st.text(raw_background if raw_background.strip() else "⚠️ EMPTY — nothing retrieved from docs")

    with st.spinner("Step 2/3 — Extracting relevant facts from your background..."):
        try:
            fact_sheet = extract_relevant_facts(raw_background, role, company, jd_text, model)
        except Exception as e:
            st.error(f"Fact extraction error: {e}")
            st.stop()

    with st.expander("Debug: Extracted fact sheet (what goes into the cover letter)"):
        st.text(fact_sheet)

    with st.spinner("Ranking your experiences against the JD..."):
        try:
            ranking = rank_relevance(fact_sheet, role, company, jd_text, model)
        except Exception as e:
            ranking = f"Ranking failed: {e}"

    with st.expander("Relevance ranking — experiences & projects vs this JD", expanded=True):
        st.markdown(ranking)

    prompt = render_prompt(
        template_name=template,
        company=company,
        role=role,
        background_context=fact_sheet,
        jd=jd_text,
        person_name=person_name,
        person_title=person_title,
        email=contact_email,
    )

    with st.expander("Prompt sent to model", expanded=preview_btn):
        st.text_area("Full prompt", value=prompt, height=400)

    if generate_btn:
        with st.spinner("Step 3/3 — Writing cover letter..."):
            try:
                result = generate(prompt, model=model)
            except RuntimeError as e:
                st.error(str(e))
                st.stop()
            except Exception as e:
                st.error(f"Generation error: {e}")
                st.stop()

        st.success("Done!")

        # Split letter and citations — try multiple possible markers
        import re
        citation_match = re.search(r'-{2,}\s*\nCITATIONS:|CITATIONS:', result, re.IGNORECASE)
        if citation_match:
            letter_part = result[:citation_match.start()].strip()
            citations_part = result[citation_match.end():].strip()
        else:
            letter_part = result.strip()
            citations_part = None

        st.markdown(letter_part)
        st.divider()
        plain = letter_part.replace("**", "")
        st.text_area("Plain text (copy this)", value=plain, height=420)
        st.download_button(
            "Download as .txt",
            data=plain,
            file_name=f"{template}_{company.lower().replace(' ', '_')}.txt",
            mime="text/plain",
        )
        if citations_part:
            with st.expander("Citations (verify grounding)"):
                st.text(citations_part)
        else:
            with st.expander("Raw output (debug — no citations found)"):
                st.text(result)
