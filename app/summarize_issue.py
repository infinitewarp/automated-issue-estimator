# summarize_issue.py
import json
import requests
from pathlib import Path
from alive_progress import alive_bar

HALLUCINATE = False  # flip this for expensive story rewriting before embedding
# HALLUCINATE = True

# OLLAMA_MODEL = "gemma3:27b" # 17GB big model
OLLAMA_MODEL = "llama3.2:3b"  # 2GB small model
OLLAMA_API_URL = "http://localhost:11434/api/generate"


def format_prompt(problem_description):
    return f"""
You are an expert product owner. Given the following problem description, generate a concise user story paragraph with one to three sentences of supporting details, starting in the form:
"As a [persona], I want [feature or behavior] so that [desired outcome]. [supporting details]."

Return only your summarized text. Do not explain additional context.

Problem description:
\"\"\"
{problem_description}
\"\"\"

User story:
"""


def generate_user_story(problem_description, force_hallucination=False):
    if not (HALLUCINATE or force_hallucination):
        return problem_description
    prompt = format_prompt(problem_description)

    payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}

    try:
        response = requests.post(OLLAMA_API_URL, json=payload)
        response.raise_for_status()
        output = response.json()["response"].strip()
        print(f"\n{output}\n")
        return output
    except requests.exceptions.RequestException as e:
        raise Exception(f"Error communicating with Ollama: {e}")
    except KeyError:
        raise Exception("Unexpected response format from Ollama.")


def rewrite_stories_json():
    stories_path = Path("stories.json")
    with stories_path.open("r") as f:
        stories = json.load(f)
    with alive_bar(len(stories)) as bar:
        for story in stories:
            story["description"] = generate_user_story(
                story["description"], force_hallucination=True
            )
            bar.text = story["id"]
            bar()
    with stories_path.open("w", encoding="utf-8") as f:
        json.dump(stories, f, indent=2)
