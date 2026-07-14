from app.core.config import PromptConfig
from app.core.schemas import GitHubIssue


def build_user_prompt(
    issue: GitHubIssue,
    prompt_config: PromptConfig,
) -> str:
    """
    Build the user prompt using the template defined
    in the YAML configuration.
    """

    return prompt_config.user_prompt_template.format(
        title=issue.title,
        body=issue.body,
    )