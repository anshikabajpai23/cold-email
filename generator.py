import os
from pathlib import Path

import requests
from jinja2 import Environment, FileSystemLoader, TemplateNotFound

TEMPLATES_DIR = Path(__file__).parent / "templates"
OLLAMA_URL = "http://localhost:11434/api/chat"


def list_templates() -> list[str]:
    return [p.stem for p in TEMPLATES_DIR.glob("*.j2")]


def _call_ollama(system: str, user: str, model: str, timeout: int = 300) -> str:
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": model,
                "stream": False,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()["message"]["content"].strip()
    except requests.exceptions.ConnectionError:
        raise RuntimeError("Ollama is not running. Start it with: ollama serve")


def extract_relevant_facts(raw_background: str, role: str, company: str, jd: str, model: str) -> str:
    system = "You are a fact extractor. Extract only real, specific facts from the provided background document. Output a clean structured list. Never invent or infer anything not explicitly stated."
    jd_section = f"Job description context:\n{jd[:800]}" if jd else ""
    user = f"""From the background document below, extract every fact relevant to the role of {role} at {company}.

Background document:
{raw_background}

{jd_section}

Output a structured list covering:
- Full name, degree, university, graduation date, years of experience (exactly as stated)
- Each work experience / internship: company name, role title, what they built, measurable impact (exact numbers only)
- Each project: project name, what it does, technologies used, outcomes or metrics (exact numbers only)
- Relevant technical skills explicitly mentioned
- Any publications, research, or notable achievements

Rules:
- Only include what is explicitly in the background document
- If a number or metric is not in the document, do not include it
- If a company name is not in the document, do not include it
- Keep it short and factual — no sentences, just facts"""

    return _call_ollama(system, user, model)


def rank_relevance(fact_sheet: str, role: str, company: str, jd: str, model: str) -> str:
    system = "You are a career advisor. Rank experiences and projects by relevance to a job description. Be concise and specific about why each is relevant or not."
    jd_section = f"Job description:\n{jd[:1000]}" if jd else f"Role: {role} at {company}"
    user = f"""Rank this candidate's experiences and projects for a specific job.

STEP 1 — Extract the top 5 technical requirements from the job description:
{jd_section}

STEP 2 — Candidate background:
{fact_sheet}

STEP 3 — Rank each work experience and project by how directly it maps to the top 5 requirements.
Ranking criteria (in order of importance):
1. Same or similar tech stack?
2. Same type of system (inference, RAG, agents, backend, CV, simulation, etc.)?
3. Measurable production impact relevant to the role?
4. Similar scale/environment?

Output:

**Top JD requirements:** (list the 5 you identified)

**WORK EXPERIENCE** (ranked most → least relevant):
1. [Company / Role] — [specific reason tied to a JD requirement], score: X/10
2. ...

**PROJECTS** (ranked most → least relevant):
1. [Project name] — [specific reason tied to a JD requirement], score: X/10
2. ...

Be precise. A production system with measurable impact ranks higher than a research project for a production role, even if the research used more advanced models."""

    return _call_ollama(system, user, model)


def render_prompt(
    template_name: str,
    company: str,
    role: str,
    background_context: str,
    jd: str = "",
    person_name: str = "",
    person_title: str = "",
    email: str = "",
) -> str:
    env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)))
    try:
        tmpl = env.get_template(f"{template_name}.j2")
    except TemplateNotFound:
        available = list_templates()
        raise ValueError(f"Template '{template_name}' not found. Available: {available}")
    return tmpl.render(
        company=company,
        role=role,
        jd=jd,
        person_name=person_name,
        person_title=person_title,
        email=email,
        background_context=background_context,
    )


def generate(prompt: str, model: str = "qwen2.5:14b") -> str:
    system = "You are a cover letter writer. Output ONLY the requested cover letter — no questions, no clarifications, no commentary. Never ask for more information. Generate immediately."
    return _call_ollama(system, prompt, model)
