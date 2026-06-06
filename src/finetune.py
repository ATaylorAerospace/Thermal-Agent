"""
QLoRA fine-tuning for open-weight models (Llama 3.3 70B / Qwen2.5 72B).

Fine-tunes a 4-bit-quantized base model with LoRA adapters on the agentic
tool-calling traces produced by ``src.training_data``. The goal is reliability,
not facts: bake the tool schemas and call format into the weights so the served
model behaves deterministically as an agent with a minimal system prompt.

Heavy dependencies (torch, transformers, peft, trl, bitsandbytes, datasets) are
imported lazily so importing this module stays cheap. Install them with::

    pip install -r requirements-finetune.txt

Run::

    python src/finetune.py --config config/finetune_config.yaml

Author: A Taylor
"""

import argparse
import logging

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path):
    """Load the fine-tuning YAML configuration.

    Args:
        config_path: Path to the config YAML.

    Returns:
        Parsed configuration dict.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_finetune(config_path="config/finetune_config.yaml"):
    """Run QLoRA fine-tuning end to end.

    Args:
        config_path: Path to the fine-tuning config YAML.

    Returns:
        The output directory containing the trained LoRA adapter.
    """
    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from trl import SFTConfig, SFTTrainer

    config = load_config(config_path)
    base_cfg = config["base_model"]
    quant_cfg = config["quantization"]
    lora_cfg = config["lora"]
    train_cfg = config["training"]

    logger.info("Loading base model: %s", base_cfg["model_id"])
    compute_dtype = getattr(torch, quant_cfg.get("bnb_4bit_compute_dtype", "bfloat16"))
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=quant_cfg.get("load_in_4bit", True),
        bnb_4bit_quant_type=quant_cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=quant_cfg.get("bnb_4bit_use_double_quant", True),
    )

    tokenizer = AutoTokenizer.from_pretrained(
        base_cfg["model_id"], trust_remote_code=base_cfg.get("trust_remote_code", False)
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_cfg["model_id"],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=base_cfg.get("trust_remote_code", False),
    )

    peft_config = LoraConfig(
        r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("alpha", 32),
        lora_dropout=lora_cfg.get("dropout", 0.05),
        target_modules=lora_cfg.get("target_modules"),
        bias=lora_cfg.get("bias", "none"),
        task_type=lora_cfg.get("task_type", "CAUSAL_LM"),
    )

    data_dir = train_cfg["data_dir"]
    dataset = load_dataset(
        "json",
        data_files={
            "train": f"{data_dir}/train.jsonl",
            "validation": f"{data_dir}/validation.jsonl",
        },
    )

    def formatting_func(example):
        """Render a chat example to text using the model's tool chat template."""
        return tokenizer.apply_chat_template(
            example["messages"],
            tools=example.get("tools"),
            tokenize=False,
        )

    sft_config = SFTConfig(
        output_dir=train_cfg["output_dir"],
        num_train_epochs=train_cfg.get("num_train_epochs", 3),
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 16),
        learning_rate=train_cfg.get("learning_rate", 2e-4),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.03),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        logging_steps=train_cfg.get("logging_steps", 10),
        save_strategy=train_cfg.get("save_strategy", "epoch"),
        bf16=train_cfg.get("bf16", True),
        max_seq_length=train_cfg.get("max_seq_length", 4096),
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
        optim=train_cfg.get("optim", "paged_adamw_8bit"),
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=peft_config,
        formatting_func=formatting_func,
    )

    logger.info("Starting QLoRA fine-tuning...")
    trainer.train()

    output_dir = train_cfg["output_dir"]
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("Adapter saved to %s", output_dir)
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="QLoRA fine-tuning for open-weight models. Author: A Taylor"
    )
    parser.add_argument("--config", type=str, default="config/finetune_config.yaml")
    args = parser.parse_args()
    run_finetune(args.config)
