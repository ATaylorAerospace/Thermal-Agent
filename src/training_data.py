"""
Build agentic tool-calling SFT data for fine-tuning open-weight models.

Closing the "out-of-the-box gap": off-the-shelf Llama 3.3 / Qwen2.5 models are
capable but raw at emitting strict tool calls for a specific stack. This module
generates supervised traces in the thermal-advisor "dialect" — each example
walks a scenario through parallel tool calls, tool results, and a grounded
final answer — so QLoRA can bake the tool schemas and behavior into the weights
and shrink the runtime system prompt.

Output is OpenAI-style chat JSONL (messages + tools) consumed by the trainer's
chat template, which both Llama 3.3 and Qwen2.5 support.

Author: A Taylor
"""

import argparse
import json
import logging
import os
from pathlib import Path

from sklearn.model_selection import train_test_split

from src.agent import SYSTEM_PROMPT
from src.backends import bedrock_tools_to_openai
from src.simulator import ENVIRONMENT_DELTA_T, MATERIAL_PROPERTIES, ThermalDriftSimulator
from src.tools import TOOL_SPECS

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

OPENAI_TOOLS = bedrock_tools_to_openai(TOOL_SPECS)
_SIMULATOR = ThermalDriftSimulator()


def _build_query(row):
    """Construct the user scenario prompt for a dataset row."""
    return (
        f"Instrument: {row.get('instrument', 'N/A')}\n"
        f"Material: {row.get('material_name', 'N/A')}\n"
        f"Environment: {row.get('environment_location', 'N/A')}\n"
        f"Thermal Effect: {row.get('thermal_effect', 'N/A')}\n"
        "What thermal mitigation strategy should be used and why?"
    )


def _simulate_result(row):
    """Compute the physics tool result, tolerating unknown inputs."""
    material = row.get("material_name")
    environment = row.get("environment_location")
    if material in MATERIAL_PROPERTIES and environment in ENVIRONMENT_DELTA_T:
        return _SIMULATOR.evaluate(material, environment)
    return {"error": "Unknown material or environment for simulator."}


def _classify_result(row, classifier=None):
    """Return classifier result; falls back to synthesized if classifier unavailable.

    Args:
        row: Scenario row mapping.
        classifier: Optional fitted StrategyClassifier instance.

    Returns:
        Dict with 'predicted_strategy' and 'probabilities'.
    """
    if classifier is not None:
        try:
            proba = classifier.predict_proba(
                row.get("material_name"),
                row.get("instrument"),
                row.get("environment_location"),
                row.get("thermal_effect"),
            )
            predicted = max(proba, key=proba.get)
            return {"predicted_strategy": predicted, "probabilities": proba}
        except Exception as exc:
            logger.debug("Classifier inference failed (%s); using synthesized fallback", exc)
    # Synthesized fallback — biased toward the labeled strategy.
    strategies = ["Passive", "Active", "Hybrid"]
    true_strategy = row.get("strategy_type", "Hybrid")
    if true_strategy not in strategies:
        true_strategy = "Hybrid"
    others = [s for s in strategies if s != true_strategy]
    proba = {true_strategy: 0.7, others[0]: 0.18, others[1]: 0.12}
    return {"predicted_strategy": true_strategy, "probabilities": proba}


def _search_result(row, datastore=None):
    """Return retrieved scenarios; falls back to synthesized if datastore unavailable.

    Args:
        row: Scenario row mapping.
        datastore: Optional built ThermalDataStore instance.

    Returns:
        Dict with 'scenarios' list.
    """
    if datastore is not None:
        try:
            query_text = (
                f"{row.get('material_name')} {row.get('instrument')} "
                f"{row.get('environment_location')} {row.get('thermal_effect')}"
            )
            scenarios = datastore.query(query_text, top_k=1)
            return {"scenarios": scenarios}
        except Exception as exc:
            logger.debug("Data store query failed (%s); using synthesized fallback", exc)
    # Synthesized fallback — single scenario derived from the row itself.
    scenario = {k: row.get(k) for k in
                ("instrument", "material_name", "environment_location", "thermal_effect")}
    scenario["strategy_type"] = row.get("strategy_type")
    scenario["similarity"] = 0.92
    return {"scenarios": [scenario]}


def _final_answer(row, sim_result):
    """Build the grounded final assistant message."""
    recommendation = row.get("strategy_recommendation")
    strategy = row.get("strategy_type", "Hybrid")
    risk = sim_result.get("risk", "elevated")
    if recommendation:
        return str(recommendation)
    return (
        f"Recommended strategy: **{strategy}**. The simulator reports {risk} thermal "
        f"risk for {row.get('material_name')} in the {row.get('environment_location')}, "
        f"and similar prior scenarios resolved {row.get('thermal_effect')} with a "
        f"{strategy.lower()} approach."
    )


def _load_or_train_classifier(df, path):
    """Load classifier from disk or train on df when path is absent.

    Args:
        df: Full scenario DataFrame (used for on-the-fly training only).
        path: Expected file path for the persisted classifier.

    Returns:
        A fitted StrategyClassifier.
    """
    from src.strategy_classifier import StrategyClassifier  # lazy — avoids heavy import at module level

    clf = StrategyClassifier()
    if Path(path).exists():
        logger.info("Loading classifier from %s", path)
        clf.load(path)
    else:
        logger.info("No classifier at %s — training on dataset (%d rows)", path, len(df))
        clf.train(df)
    return clf


def _load_or_build_datastore(df, path):
    """Load data store from disk or build from df when path is absent.

    Args:
        df: Full scenario DataFrame (used for on-the-fly building only).
        path: Expected file path for the persisted data store.

    Returns:
        A built ThermalDataStore.
    """
    from src.datastore import ThermalDataStore  # lazy — avoids heavy import at module level

    store = ThermalDataStore()
    if Path(path).exists():
        logger.info("Loading data store from %s", path)
        store.load(path)
    else:
        logger.info("No data store at %s — building from dataset (%d rows)", path, len(df))
        store.build(df)
    return store


def build_example(row, classifier=None, datastore=None):
    """Build a single SFT example (messages + tools) for a dataset row.

    Args:
        row: A mapping with scenario fields and a labeled strategy.
        classifier: Optional fitted StrategyClassifier for real classify_strategy results.
        datastore: Optional built ThermalDataStore for real search_thermal_knowledge results.

    Returns:
        Dict with 'messages' (OpenAI chat format) and 'tools'.
    """
    sim_result = _simulate_result(row)
    classify_res = _classify_result(row, classifier=classifier)
    search_res = _search_result(row, datastore=datastore)

    tool_calls = [
        {
            "id": "call_sim",
            "type": "function",
            "function": {
                "name": "simulate_thermal_drift",
                "arguments": json.dumps({
                    "material": row.get("material_name"),
                    "environment": row.get("environment_location"),
                }),
            },
        },
        {
            "id": "call_clf",
            "type": "function",
            "function": {
                "name": "classify_strategy",
                "arguments": json.dumps({
                    "material": row.get("material_name"),
                    "instrument": row.get("instrument"),
                    "environment": row.get("environment_location"),
                    "thermal_effect": row.get("thermal_effect"),
                }),
            },
        },
        {
            "id": "call_kb",
            "type": "function",
            "function": {
                "name": "search_thermal_knowledge",
                "arguments": json.dumps({
                    "query": f"{row.get('material_name')} {row.get('instrument')} "
                             f"{row.get('environment_location')} {row.get('thermal_effect')}",
                    "top_k": 1,
                }),
            },
        },
    ]

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _build_query(row)},
        {"role": "assistant", "content": "", "tool_calls": tool_calls},
        {"role": "tool", "tool_call_id": "call_sim",
         "content": json.dumps(sim_result)},
        {"role": "tool", "tool_call_id": "call_clf",
         "content": json.dumps(classify_res)},
        {"role": "tool", "tool_call_id": "call_kb",
         "content": json.dumps(search_res)},
        {"role": "assistant", "content": _final_answer(row, sim_result)},
    ]
    return {"messages": messages, "tools": OPENAI_TOOLS}


def build_examples(df, limit=None, classifier=None, datastore=None):
    """Build SFT examples for a DataFrame of scenarios.

    Args:
        df: DataFrame of scenario rows.
        limit: Optional cap on the number of examples.
        classifier: Optional fitted StrategyClassifier.
        datastore: Optional built ThermalDataStore.

    Returns:
        List of example dicts.
    """
    if limit is not None:
        df = df.head(limit)
    examples = [
        build_example(row, classifier=classifier, datastore=datastore)
        for row in df.to_dict(orient="records")
    ]
    logger.info("Built %d SFT examples", len(examples))
    return examples


def write_jsonl(examples, path):
    """Write examples to a JSONL file.

    Args:
        examples: List of example dicts.
        path: Output file path.

    Returns:
        The Path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    logger.info("Wrote %d examples to %s", len(examples), path)
    return path


def run(
    output_dir="data/finetune",
    val_ratio=0.1,
    limit=None,
    dataset_name=None,
    classifier_path=None,
    index_path=None,
):
    """Build train/validation SFT JSONL from the HuggingFace dataset.

    Loads (or trains on-the-fly) a StrategyClassifier and (or builds on-the-fly)
    a ThermalDataStore so that tool-result messages contain real model outputs
    rather than synthesized proxies.

    Args:
        output_dir: Directory for the output JSONL files.
        val_ratio: Validation split fraction.
        limit: Optional cap on rows processed.
        dataset_name: HuggingFace dataset id (defaults to env/project default).
        classifier_path: Path to a saved StrategyClassifier pickle.
        index_path: Path to a saved ThermalDataStore pickle.
    """
    from datasets import load_dataset  # optional/heavy dependency

    name = dataset_name or os.getenv(
        "HF_DATASET", "Taylor658/deep-space-optical-chip-thermal-dataset"
    )
    logger.info("Loading dataset from HuggingFace: %s", name)
    df = load_dataset(name, split="train").to_pandas()
    if limit is not None:
        df = df.head(limit)

    clf_path = classifier_path or os.getenv("CLASSIFIER_PATH", "results/strategy_classifier.pkl")
    ds_path = index_path or os.getenv("INDEX_PATH", "results/thermal_datastore.pkl")

    classifier = _load_or_train_classifier(df, clf_path)
    datastore = _load_or_build_datastore(df, ds_path)

    stratify = df["strategy_type"] if "strategy_type" in df.columns else None
    train_df, val_df = train_test_split(
        df, test_size=val_ratio, random_state=42, stratify=stratify
    )

    output_dir = Path(output_dir)
    write_jsonl(
        build_examples(train_df, classifier=classifier, datastore=datastore),
        output_dir / "train.jsonl",
    )
    write_jsonl(
        build_examples(val_df, classifier=classifier, datastore=datastore),
        output_dir / "validation.jsonl",
    )
    logger.info("Training data preparation complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build agentic SFT data for open-weight fine-tuning. Author: A Taylor"
    )
    parser.add_argument("--output_dir", type=str, default="data/finetune")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--limit", type=int, default=None, help="Cap rows (for quick runs)")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--classifier_path", type=str, default=None,
                        help="Path to saved StrategyClassifier pickle")
    parser.add_argument("--index_path", type=str, default=None,
                        help="Path to saved ThermalDataStore pickle")
    args = parser.parse_args()

    run(
        output_dir=args.output_dir,
        val_ratio=args.val_ratio,
        limit=args.limit,
        dataset_name=args.dataset,
        classifier_path=args.classifier_path,
        index_path=args.index_path,
    )
