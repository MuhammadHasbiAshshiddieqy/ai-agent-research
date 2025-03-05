from github import Github

def fetch_pr_diff(token: str, repo_owner: str, repo_name: str, pr_number: int):
    # Authenticate using GitHub token
    g = Github(token)

    # Access the repository and the pull request
    repo = g.get_repo(f"{repo_owner}/{repo_name}")
    pr = repo.get_pull(pr_number)

    # Get the diff of the pull request
    files = pr.get_files()

    # Iterate through the files and print their diffs
    for file in files:
        print(f"File: {file.filename}")
        print(f"Patch:\n{file.patch}")
