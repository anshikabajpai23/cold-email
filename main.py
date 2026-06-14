from pathlib import Path
from typing import Optional

import typer
from dotenv import load_dotenv

load_dotenv()

from generator import generate, list_templates, render_prompt
from rag import build_index, query_background

app = typer.Typer(help="Cover letter & outreach generator powered by RAG.")


@app.command()
def index():
    """Rebuild the vector index from docs/."""
    typer.echo("Indexing docs...")
    build_index()
    typer.echo("Done. Index saved to cache/.")


@app.command()
def templates():
    """List available prompt templates."""
    available = list_templates()
    if not available:
        typer.echo("No templates found in templates/.")
    else:
        typer.echo("Available templates:")
        for t in available:
            typer.echo(f"  {t}")


@app.command()
def generate_cmd(
    company: str = typer.Option(..., "--company", help="Company name"),
    role: str = typer.Option(..., "--role", help="Job role / title"),
    jd: Optional[Path] = typer.Option(None, "--jd", help="Path to job description text file"),
    person_name: Optional[str] = typer.Option(None, "--person-name", help="Recipient name"),
    person_title: Optional[str] = typer.Option(None, "--person-title", help="Recipient title (Recruiter, Hiring Manager, etc.)"),
    email: Optional[str] = typer.Option(None, "--email", help="Recipient email"),
    template: str = typer.Option("cover_letter", "--template", help="Template name (without .j2)"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Save output to file"),
    model: str = typer.Option("gpt-4o", "--model", help="OpenAI model"),
):
    """Generate a cover letter or outreach message."""
    jd_text = jd.read_text() if jd else ""

    rag_query = f"{role} at {company}. {jd_text[:500]}"
    typer.echo("Retrieving relevant background context...")
    background_context = query_background(rag_query)

    prompt = render_prompt(
        template_name=template,
        company=company,
        role=role,
        background_context=background_context,
        jd=jd_text,
        person_name=person_name or "",
        person_title=person_title or "",
        email=email or "",
    )

    typer.echo(f"Generating with {model}...\n")
    result = generate(prompt, model=model)

    typer.echo(result)

    if output:
        output.write_text(result)
        typer.echo(f"\nSaved to {output}")


app.command("generate")(generate_cmd)

if __name__ == "__main__":
    app()
