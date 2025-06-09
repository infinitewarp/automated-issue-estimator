# AI/LLM/ML-based issue-estimating tool

See also: [DISCOVERY-942](https://issues.redhat.com/browse/DISCOVERY-942).

## Usage

```sh
uv sync
uv run python3 ./jira_extractor.py > stories.json
uv run python3 ./train_embeddings.py
uv run python3 ./estimate_size.py "As a Discovery user, I want you to sudo make me a sandwich."
```
