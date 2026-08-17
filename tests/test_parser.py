import unittest
from pydantic import ValidationError
from app.llm.parser import parse_llm_response
from app.core.schemas import IssueClassification


class TestLLMParser(unittest.TestCase):
    def test_parse_clean_json(self):
        raw = '{"category": "bug", "priority": "high", "summary": "Login page fails to render."}'
        result = parse_llm_response(raw)
        self.assertIsInstance(result, IssueClassification)
        self.assertEqual(result.category, "bug")
        self.assertEqual(result.priority, "high")
        self.assertEqual(result.summary, "Login page fails to render.")

    def test_parse_markdown_json_fences(self):
        raw = '```json\n{"category": "feature-request", "priority": "medium", "summary": "Add dark mode to user settings."}\n```'
        result = parse_llm_response(raw)
        self.assertIsInstance(result, IssueClassification)
        self.assertEqual(result.category, "feature-request")
        self.assertEqual(result.priority, "medium")

    def test_parse_invalid_json_raises_error(self):
        raw = '{"category": "bug", "priority": "high", "summary": "Forgot closing brace"'
        with self.assertRaises(Exception):
            parse_llm_response(raw)

    def test_parse_invalid_fields_raises_validation_error(self):
        # invalid category value 'unknown' should fail literal validation
        raw = '{"category": "unknown", "priority": "high", "summary": "Category is invalid."}'
        with self.assertRaises(ValidationError):
            parse_llm_response(raw)


if __name__ == "__main__":
    unittest.main()
