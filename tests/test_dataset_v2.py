import json
import re
import unittest
from pathlib import Path

from app.evaluation.schema import GoldenDataset


class TestGoldenDatasetV2(unittest.TestCase):
    """Automated validation tests for Golden Dataset V2 and V1 backward compatibility."""

    SUPPORTED_CATEGORIES = {
        "authentication",
        "ui",
        "backend",
        "database",
        "performance",
        "documentation",
        "bug",
        "feature-request",
    }
    SUPPORTED_PRIORITIES = {"low", "medium", "high"}
    SUPPORTED_DIFFICULTIES = {"easy", "medium", "hard"}

    @classmethod
    def setUpClass(cls):
        cls.v1_path = Path("datasets/v1/github_issues.json")
        cls.v2_path = Path("datasets/v2/github_issues.json")

        with open(cls.v1_path, "r", encoding="utf-8") as f:
            cls.v1_raw = json.load(f)
        with open(cls.v2_path, "r", encoding="utf-8") as f:
            cls.v2_raw = json.load(f)

        cls.v1_dataset = GoldenDataset.model_validate(cls.v1_raw)
        cls.v2_dataset = GoldenDataset.model_validate(cls.v2_raw)

    def test_v1_integrity_preserved(self):
        """Verify original V1 dataset has not been modified or corrupted."""
        self.assertEqual(self.v1_dataset.version, "v1")
        self.assertEqual(len(self.v1_dataset.cases), 5)
        case_ids = [c.id for c in self.v1_dataset.cases]
        self.assertEqual(case_ids, ["GH001", "GH002", "GH003", "GH004", "GH005"])

        # Check exact fields of GH001
        c1 = self.v1_dataset.cases[0]
        self.assertEqual(c1.id, "GH001")
        self.assertEqual(c1.expected.category, "bug")
        self.assertEqual(c1.expected.priority, "high")
        self.assertEqual(c1.metadata.difficulty, "easy")

    def test_v2_pydantic_schema_compliance(self):
        """Verify V2 dataset parses completely through the Pydantic GoldenDataset schema."""
        self.assertEqual(self.v2_dataset.version, "v2")
        self.assertIn("50", self.v2_dataset.description)
        self.assertEqual(len(self.v2_dataset.cases), 50)

    def test_v2_unique_case_ids(self):
        """Verify every case in V2 has a unique, non-colliding ID following GHxxx format."""
        ids = [c.id for c in self.v2_dataset.cases]
        self.assertEqual(len(ids), len(set(ids)), "All case IDs in V2 must be unique")
        id_pattern = re.compile(r"^GH\d{3}$")
        for cid in ids:
            self.assertTrue(id_pattern.match(cid), f"Case ID '{cid}' does not match format GHxxx")

    def test_v2_v1_cases_preserved_as_subset(self):
        """Verify cases GH001-GH005 in V2 match the original V1 cases identically."""
        for i in range(5):
            v1_c = self.v1_dataset.cases[i]
            v2_c = self.v2_dataset.cases[i]
            self.assertEqual(v1_c.id, v2_c.id)
            self.assertEqual(v1_c.input.title, v2_c.input.title)
            self.assertEqual(v1_c.input.body, v2_c.input.body)
            self.assertEqual(v1_c.expected.category, v2_c.expected.category)
            self.assertEqual(v1_c.expected.priority, v2_c.expected.priority)
            self.assertEqual(v1_c.expected.summary, v2_c.expected.summary)
            self.assertEqual(v1_c.metadata.difficulty, v2_c.metadata.difficulty)
            self.assertEqual(v1_c.metadata.tags, v2_c.metadata.tags)
            self.assertEqual(v1_c.metadata.notes, v2_c.metadata.notes)

    def test_v2_category_taxonomy_and_coverage(self):
        """Verify all cases use valid categories and every supported category is represented."""
        observed_categories = set()
        category_counts = {}

        for c in self.v2_dataset.cases:
            cat = c.expected.category
            self.assertIn(cat, self.SUPPORTED_CATEGORIES, f"Case {c.id} has invalid category: '{cat}'")
            observed_categories.add(cat)
            category_counts[cat] = category_counts.get(cat, 0) + 1

        self.assertEqual(
            observed_categories,
            self.SUPPORTED_CATEGORIES,
            "All 8 supported categories must be represented in V2",
        )
        for cat, count in category_counts.items():
            self.assertGreaterEqual(
                count, 6, f"Category '{cat}' has only {count} cases; expected balanced representation (>=6)"
            )

    def test_v2_priority_taxonomy_and_coverage(self):
        """Verify all cases use valid priorities and all 3 priority levels are well-represented."""
        observed_priorities = set()
        priority_counts = {}

        for c in self.v2_dataset.cases:
            pri = c.expected.priority
            self.assertIn(pri, self.SUPPORTED_PRIORITIES, f"Case {c.id} has invalid priority: '{pri}'")
            observed_priorities.add(pri)
            priority_counts[pri] = priority_counts.get(pri, 0) + 1

        self.assertEqual(
            observed_priorities,
            self.SUPPORTED_PRIORITIES,
            "All 3 priority levels (low, medium, high) must be represented in V2",
        )
        for pri, count in priority_counts.items():
            self.assertGreaterEqual(
                count, 10, f"Priority '{pri}' has only {count} cases; expected balanced representation (>=10)"
            )

    def test_v2_difficulty_taxonomy_and_coverage(self):
        """Verify all cases use valid difficulties and all 3 difficulty tiers are represented."""
        observed_difficulties = set()
        difficulty_counts = {}

        for c in self.v2_dataset.cases:
            diff = c.metadata.difficulty
            self.assertIn(diff, self.SUPPORTED_DIFFICULTIES, f"Case {c.id} has invalid difficulty: '{diff}'")
            observed_difficulties.add(diff)
            difficulty_counts[diff] = difficulty_counts.get(diff, 0) + 1

        self.assertEqual(
            observed_difficulties,
            self.SUPPORTED_DIFFICULTIES,
            "All 3 difficulty tiers (easy, medium, hard) must be represented in V2",
        )
        for diff, count in difficulty_counts.items():
            self.assertGreaterEqual(
                count, 8, f"Difficulty '{diff}' has only {count} cases; expected adequate representation (>=8)"
            )

    def test_v2_input_integrity_and_non_emptiness(self):
        """Verify no empty or whitespace-only titles, bodies, summaries, or notes exist."""
        for c in self.v2_dataset.cases:
            self.assertTrue(c.input.title and c.input.title.strip(), f"Case {c.id} has empty title")
            self.assertTrue(c.input.body and c.input.body.strip(), f"Case {c.id} has empty body")
            self.assertTrue(c.expected.summary and c.expected.summary.strip(), f"Case {c.id} has empty summary")
            self.assertTrue(c.metadata.notes and c.metadata.notes.strip(), f"Case {c.id} has empty notes")

            # Check that description is sufficiently substantive
            self.assertGreaterEqual(
                len(c.input.body),
                30,
                f"Case {c.id} body too short ({len(c.input.body)} chars); must be realistic description",
            )

    def test_v2_tags_integrity(self):
        """Verify tags are non-empty lists of lowercase string identifiers."""
        for c in self.v2_dataset.cases:
            tags = c.metadata.tags
            self.assertIsInstance(tags, list, f"Case {c.id} tags must be a list")
            self.assertGreaterEqual(len(tags), 1, f"Case {c.id} must have at least 1 tag")
            self.assertEqual(len(tags), len(set(tags)), f"Case {c.id} has duplicate tags")
            for t in tags:
                self.assertTrue(t and t.strip(), f"Case {c.id} has empty tag")
                self.assertEqual(t, t.lower(), f"Case {c.id} tag '{t}' must be lowercase")


if __name__ == "__main__":
    unittest.main()
