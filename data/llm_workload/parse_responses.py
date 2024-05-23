from pathlib import Path
from typing import Literal

from fainder.typing import PercentileQuery
from fainder.utils import save_output

if __name__ == "__main__":
    with open(Path(__file__).parent / "responses.txt", "r") as f:
        lines = f.readlines()

    predicates: dict[str, list[PercentileQuery]] = {}
    for line in lines:
        if not line or not line.strip():
            continue
        if line.startswith("#"):
            model = line[1:].strip()
            predicates[model] = []
            continue
        elements = line.strip()[1:-1].split(", ")

        try:
            assert len(elements) == 3
            percentile = float(elements[1])
            if percentile <= 0 or percentile > 1:
                raise ValueError(f"Invalid percentile: {percentile}")
            if elements[0] not in ["<", "<=", ">", ">="]:
                raise ValueError(f"Invalid comparison: {elements[0]}")
            comparison: Literal["gt", "ge", "lt", "le"]
            match elements[0]:
                # NOTE: We have to rewrite the queries due to the differences between our
                # formalization and implementation.
                case "<":
                    comparison = "gt"
                    percentile = 1 - percentile
                case "<=":
                    comparison = "ge"
                    percentile = 1 - percentile
                case ">":
                    comparison = "lt"
                case ">=":
                    comparison = "le"
            reference = float(elements[2])
        except Exception as e:
            raise ValueError(f"Error in line: {line}") from e

        predicates[model].append((percentile, comparison, reference))

    # Duplicate check
    unique_predicates = set()
    for model, preds in predicates.items():
        for pred in preds:
            if pred in unique_predicates:
                predicates[model].remove(pred)
            unique_predicates.add(pred)

    # Sample at most 50 predicates per model
    for model, preds in predicates.items():
        print(f"{model}: {len(preds)}")
        if model == "gemini-1.5-pro-api-0409-preview" and len(preds) > 100:
            predicates[model] = preds[:100]
        elif model == "gpt-4-turbo-2024-04-09" and len(preds) > 150:
            predicates[model] = preds[:150]
        elif len(preds) > 50:
            predicates[model] = preds[:50]

    # Flatten all predicates into a list and verify the length
    predicate_list = [pred for preds in predicates.values() for pred in preds]
    assert len(predicate_list) == 500

    save_output(Path(__file__).parent / "llm_queries.zst", predicate_list, "LLM queries")
