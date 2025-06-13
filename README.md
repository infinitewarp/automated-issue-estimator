# AI/LLM/ML-based issue-estimating tool

This is a AI/LLM/ML-powered command-line program that takes a user story description as input and generates an estimated size for that story and a list of the most related stories as output. Although this program is merely a **proof-of-concept prototype**, its functionality demonstrates the potential for language model tools to assist in the agile story-building processes.

![](readme-demo.webp)

## Background Info

I built this prototype to satisfy a 2025Q2 department-wide goal of experimenting with AI/LLM/ML-related technologies, and I gave a live demo to my team on 2025-06-13.

The original intent and scope was defined in this user story: [DISCOVERY-942](https://issues.redhat.com/browse/DISCOVERY-942).

> As Brad, I want to try to build an AI/LLM/ML-based issue-estimating tool so that I can experiment with ML-related technologies under the broader RH AI mandate. This tool will serve primarily as a platform to experiment with a few ML-related technologies, and although I am hopeful that I can make it produce reasonable outputs, I do not expect or intend for this to be productized into a solution that Discovery or other Red Hat teams will use going forward.
>
> *Acceptance Criteria*
>
> * Build a program that takes a user story description as input and outputs an estimated size tag (small, medium, large) as its output.
>     * Stretch goal: output some similar previously-defined user stories.
> * Assuming it functions, demonstrate the program to the Discovery team.
> * Explain the technologies chosen, why they were chosen, and any challenges they posed.
>
> I (Brad) think this will include the following operations:
>
> * Scrape issue descriptions and labels from Jira, and store locally to a flat file or database.
> * Use a small model to build embeddings for each issue, and train a classifier on them.
> * Structure the issues into a format that can be loaded for RAG (retrieval-augmented generation).
> * Write a program that interfaces with the classifier and an LLM with the RAG so that:
>     * the classifier outputs an estimated size
>     * the LLM+RAG outputs a short list of issues that seem similar
>
> *Assumptions and Questions*
>
> * My assumptions about embeddings and RAG are based on some light research, not experience. I could be completely wrong about their application to this problem.
> * Output is a prototype, not a production-ready application or service.
> * Interface will likely be a local CLI or shell script.
> * Demo may have to be prerecorded depending on availability and performance of hardware.

## PoC WARNING DISCLAIMER

This code and this repo represent a very quick one-time coding effort to assemble a **proof-of-concept prototype** tool to assist in estimating user stories during my software engineering team's periodic backlog refiment (a.k.a. "grooming") ceremonies.

### The code here is not production quality.

**Do not use this directly.** This project may serve as inspiration for future work or the basis for a production-ready product only after much more iteration.

## Prereqs

* Python and uv
* network access
    * to install python packages
    * to pull models from 🤗 Hugging Face
* optionally Ollama
    * to preprocess summarize descriptions into a more normal form
    * also needs network access to pull models
* Jira credentials or personal access token
    * if you want to scrape issues
    * `~/.jirasucks.json` looks like:
    ```
    {"prod": ["username", "yourpersonalaccessgoestokenhere"]}
    ```
* Local hardware that is capable of LLM training and inference.

## Usage

```sh
uv sync
uv run python app/cli.py getjira
uv run python app/cli.py rewrite
uv run python app/cli.py train
uv run python app/cli.py predict --text "As a Discovery user, I want you to sudo make me a sandwich."
uv run python app/cli.py predict --repl --rewrite
```
