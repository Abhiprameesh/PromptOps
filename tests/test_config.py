import unittest
from pathlib import Path
from app.core.config import load_prompt_config, PromptConfig


class TestConfigLoading(unittest.TestCase):
    def test_load_valid_config(self):
        # We target the default prompt config in prompt/v1.yaml
        config_path = Path("prompt") / "v1.yaml"
        self.assertTrue(config_path.exists(), "prompt/v1.yaml does not exist")

        config = load_prompt_config(str(config_path))

        self.assertIsInstance(config, PromptConfig)
        self.assertEqual(config.version, "v1")
        self.assertEqual(config.name, "GitHub Issue Triage")
        self.assertEqual(config.model, "gemma3:4b")
        self.assertEqual(config.temperature, 0.0)
        self.assertTrue(hasattr(config, "system_prompt"))
        self.assertTrue(hasattr(config, "user_prompt_template"))
        self.assertIn("category", config.output_schema)
        self.assertIn("priority", config.output_schema)
        self.assertIn("summary", config.output_schema)


if __name__ == "__main__":
    unittest.main()
