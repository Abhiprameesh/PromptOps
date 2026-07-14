from app.core.config import load_prompt_config
from app.core.schemas import GitHubIssue
from app.llm.formatter import build_user_prompt


config = load_prompt_config("prompt/v1.yaml")

issue = GitHubIssue(
    title="Cannot login",
    body="After resetting my password I cannot login anymore.",
)

prompt = build_user_prompt(issue, config)

print("=" * 50)
print("Formatted Prompt")
print("=" * 50)
print(prompt)