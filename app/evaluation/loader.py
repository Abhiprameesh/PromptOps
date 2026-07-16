import json
from pathlib import Path

from app.evaluation.schema import GoldenDataset


def load_dataset(path: str) -> GoldenDataset:
    """
    Load and validate the evaluation dataset.
    """

    with open(Path(path), "r", encoding="utf-8") as file:
        data = json.load(file)

    return GoldenDataset(**data)