# cli.py
import argparse
from estimate_size import estimate_size
from train_embeddings import train_embeddings
from predict_repl import predict_repl
from jira_downloader import download


def predict_once(story_text):
    print()
    estimate_size(story_text)


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

    args = parser.parse_args()

    if args.command == "getjira":
        download()
    elif args.command == "rewrite":
        train_embeddings()
    elif args.command == "train":
        train_embeddings()
    elif args.command == "predict":
        if args.repl:
            predict_repl()
        elif args.text:
            predict_once(args.text)
        else:
            print("Please provide --text or --repl for predict.")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
