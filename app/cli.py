# cli.py
import argparse
from jira_downloader import download
from summarize_issue import rewrite_stories_json
from train_embeddings import train_embeddings
from estimate_size import estimate_size
from predict_repl import predict_repl


def predict_once(story_text, force_hallucination=False):
    print()
    estimate_size(story_text, force_hallucination)


def main():
    parser = argparse.ArgumentParser(description="Issue Size Estimator")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("getjira", help="Pull data from Jira and write stories.json")
    subparsers.add_parser("rewrite", help="Rewrite stories.json with LLM magic")

    subparsers.add_parser("train", help="Train model from stories.json")
    # subparsers.add_parser("update", help="Update model with new_stories.json")
    predict_parser = subparsers.add_parser(
        "predict", help="Predict size for a user story"
    )
    predict_parser.add_argument(
        "--repl", action="store_true", help="Run in interactive REPL mode"
    )
    predict_parser.add_argument(
        "--text", type=str, help="Single prediction for given story text"
    )
    predict_parser.add_argument(
        "--rewrite", action="store_true", help="Rewrite input with LLM magic"
    )

    args = parser.parse_args()

    if args.command == "getjira":
        download()
    elif args.command == "rewrite":
        rewrite_stories_json()
    elif args.command == "train":
        train_embeddings()
    elif args.command == "predict":
        if args.repl:
            predict_repl(args.rewrite)
        elif args.text:
            predict_once(args.text, args.rewrite)
        else:
            print("Please provide --text or --repl for predict.")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
