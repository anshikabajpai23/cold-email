# Cold Email / Cover Letter Generator

AI-powered personalized cover letter and outreach generator using RAG over Anshika's background documents.

---

## Project Structure

```
cold-email/
├── docs/                  # Drop resume, bio, project writeups here
├── templates/
│   ├── cover_letter.j2    # Cover letter prompt template
│   └── linkedin_message.j2
├── cache/                 # Auto-created vector store
├── app.py                 # Streamlit web UI (primary interface)
├── main.py                # CLI (Typer): generate, index, templates
├── rag.py                 # LlamaIndex RAG pipeline
├── generator.py           # Template render + OpenAI call
├── .env                   # OPENAI_API_KEY (not committed)
└── requirements.txt
```

## Running the App

### Web UI (recommended)
```bash
streamlit run app.py
```
Opens at http://localhost:8501. Upload docs, build index, fill the form, hit Generate.

### CLI
```bash
# Index background docs (run once, or when docs change)
python main.py index

# Generate a cover letter
python main.py generate \
  --company "Stripe" \
  --role "Product Manager" \
  --jd jd.txt \
  --person-name "Sarah" \
  --person-title "Recruiter" \
  --template cover_letter \
  --output out.txt

# List available templates
python main.py templates
```

## Setup
```bash
pip install -r requirements.txt
cp .env.example .env   # add OPENAI_API_KEY
```

## Cover Letter Requirements

### Content priorities (in order)
1. Relevant work experience / internships first
2. Personal projects with high impact (relevant to the role)
3. Unique skillset — what Anshika brings that others don't

### Tone & language
- Simple, conversational English — sounds like a real person, not AI
- NO generic phrases: "what excites me about...", "I am passionate about...", "I would love to..."
- Show genuine interest naturally — through specific knowledge, not Google-searched facts
- Mention something real about the company that a normal curious person would find interesting (an idea, a product decision, a space they operate in) — NOT trivia or PR talking points

### Structure (strict)
- **Para 1**: Hook + relevant experience/projects + unique value proposition
- **Para 2**: How Anshika contributes to THIS company specifically + soft skills relevant to role (ownership, impact, fast-paced, leadership for senior roles, large-scale thinking for big companies)
- **Para 3 (2–3 lines max)**: Clean close with contact info
  - Phone: +1 (930) 204-3030
  - Email: anshikabajpai23@gmail.com

### Values to reflect (match to role/company size)
- Startups: ownership, speed, impact, wearing many hats
- Big companies: scale, cross-functional leadership, systems thinking

## Adding Templates

Add a new `.j2` file under `templates/`. Variables available in every template:
- `{{ company }}`, `{{ role }}`, `{{ jd }}` (may be empty string)
- `{{ person_name }}`, `{{ person_title }}`, `{{ email }}` (may be empty string)
- `{{ background_context }}` — RAG-retrieved chunks from docs/

## Adding Background Docs

Drop any `.txt`, `.pdf`, or `.md` file into `docs/` and re-run `python main.py index`.
