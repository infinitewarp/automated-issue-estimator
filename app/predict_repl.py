from estimate_size import estimate_size

from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory

session = PromptSession(
    multiline=True, history=InMemoryHistory(), auto_suggest=AutoSuggestFromHistory()
)


def predict_repl(force_hallucination=False):
    print(
        "\n📘 Enter a user story description. Submit with ESCAPE followed by ENTER. Ctrl+C to exit.\n"
    )
    while True:
        try:
            text = session.prompt(">>> ")
            if text.strip():
                estimate_size(text, force_hallucination)
                print()
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Exiting REPL.")
            break
