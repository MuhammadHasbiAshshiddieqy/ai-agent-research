import os

repo_name = os.getenv('REPO_NAME', 'default-repo-name')
pr_number = os.getenv('PR_NUMBER', 'no-pr')

print(f"Repository Name: {repo_name}")
print(f"Pull Request Number: {pr_number}")
