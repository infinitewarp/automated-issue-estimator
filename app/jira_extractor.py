# jira_extractor
import json
from getpass import getpass, getuser
from os.path import expanduser

from jira import JIRA

import os


SERVER = "https://issues.redhat.com"


def get_auth():
    env_username = os.environ.get("JIRA_PROD_USERNAME")
    env_password = os.environ.get("JIRA_PROD_PASSWORD")

    try:
        with open(expanduser("~/.jirasucks.json"), "rt") as fd:
            data = json.load(fd)
            conf_username, conf_password = data.get("prod", (None, None))
    except Exception:
        # yup, suppressing problems to fall back to nothingness
        conf_username, conf_password = (None, None)

    auth_username = env_username or conf_username or getuser()
    auth_password = env_password or conf_password \
        or getpass("Jira Personal Access Token: ")

    return (auth_username, auth_password)


class Jira(JIRA):

    def __init__(self):
        super(Jira, self).__init__(SERVER)

        # inject our bearer token as a permanent header
        self._session.headers["Authorization"] = "Bearer " + get_auth()[1]
        self._customfields = None


size_labels = {"small", "medium", "large", "xlarge"}
jql_query = f"""
    project=DISCOVERY AND labels in {tuple(size_labels)}
    ORDER BY created DESC
"""

def get_issue_data():
    start_at = 0
    max_results = 200
    all_issues = []

    jira = Jira()
    while True:
        issues = jira.search_issues(
            jql_str=jql_query,
            startAt=start_at,
            maxResults=max_results,
            fields="summary,description,labels"
        )

        if not issues:
            break

        # issues = issues[::]
        all_issues.extend(issues)
        start_at += len(issues)

        if len(issues) < max_results:
            break  # Last page reached

    results = []

    for issue in all_issues:
        issue_labels = set(issue.fields.labels)
        size = next((label for label in size_labels if label in issue_labels), None)

        results.append({
            "id": issue.key,
            "title": issue.fields.summary,
            "description": issue.fields.description or "",
            "size": size
        })
    
    return results

if __name__ == "__main__":
    print(json.dumps(get_issue_data(), indent=2))
