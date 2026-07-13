from app.core.config import load_prompt_config

config = load_prompt_config(r"C:\ALLMLPROJ\MAINMLPROJS\PromptOps\prompt\v1.yaml")

print("=" * 50)
print("Prompt Configuration Loaded Successfully")
print("=" * 50)

print(config.model_dump_json(indent=4))