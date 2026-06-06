"""
Merge a QLoRA adapter into its base model and export to quantized GGUF.

Two stages:
  1. Merge the trained LoRA adapter into the base weights (peft) and save a
     standalone HuggingFace model.
  2. Convert that model to GGUF and quantize it with llama.cpp so it can be
     served cheaply on CPU/GPU via llama.cpp, Ollama, or vLLM.

Heavy dependencies (torch, transformers, peft) and the llama.cpp tools are used
lazily / via subprocess, so importing this module stays cheap.

Run::

    python src/quantize.py --config config/finetune_config.yaml

Author: A Taylor
"""

import argparse
import logging
import subprocess
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path):
    """Load the fine-tuning/quantization YAML configuration.

    Args:
        config_path: Path to the config YAML.

    Returns:
        Parsed configuration dict.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def merge_adapter(base_model_id, adapter_dir, merged_dir, trust_remote_code=False):
    """Merge a LoRA adapter into the base model and save the result.

    Args:
        base_model_id: HuggingFace id of the base model.
        adapter_dir: Directory of the trained LoRA adapter.
        merged_dir: Output directory for the merged model.
        trust_remote_code: Pass-through to the model/tokenizer loaders.

    Returns:
        The merged model directory as a Path.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("Loading base model for merge: %s", base_model_id)
    base = AutoModelForCausalLM.from_pretrained(
        base_model_id, torch_dtype=torch.float16, trust_remote_code=trust_remote_code
    )
    logger.info("Applying adapter: %s", adapter_dir)
    model = PeftModel.from_pretrained(base, adapter_dir)
    model = model.merge_and_unload()

    merged_dir = Path(merged_dir)
    merged_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(merged_dir)
    AutoTokenizer.from_pretrained(
        base_model_id, trust_remote_code=trust_remote_code
    ).save_pretrained(merged_dir)
    logger.info("Merged model saved to %s", merged_dir)
    return merged_dir


def export_gguf(merged_dir, output_path, quant_type, llama_cpp_dir):
    """Convert a merged model to GGUF and quantize it with llama.cpp.

    Args:
        merged_dir: Directory of the merged HuggingFace model.
        output_path: Final quantized GGUF path.
        quant_type: llama.cpp quantization type (e.g. Q4_K_M).
        llama_cpp_dir: Path to a built llama.cpp checkout.

    Returns:
        The quantized GGUF path as a Path.
    """
    llama_cpp_dir = Path(llama_cpp_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    convert_script = llama_cpp_dir / "convert_hf_to_gguf.py"
    f16_path = output_path.with_suffix(".f16.gguf")

    logger.info("Converting to GGUF (f16): %s", f16_path)
    subprocess.run(
        ["python", str(convert_script), str(merged_dir),
         "--outfile", str(f16_path), "--outtype", "f16"],
        check=True,
    )

    quantize_bin = _find_quantize_binary(llama_cpp_dir)
    logger.info("Quantizing to %s: %s", quant_type, output_path)
    subprocess.run(
        [str(quantize_bin), str(f16_path), str(output_path), quant_type],
        check=True,
    )
    logger.info("Quantized GGUF written to %s", output_path)
    return output_path


def _find_quantize_binary(llama_cpp_dir):
    """Locate the llama.cpp quantize binary in a checkout.

    Args:
        llama_cpp_dir: Path to a built llama.cpp checkout.

    Returns:
        Path to the quantize binary.

    Raises:
        FileNotFoundError: If no quantize binary is found.
    """
    candidates = [
        llama_cpp_dir / "llama-quantize",
        llama_cpp_dir / "build" / "bin" / "llama-quantize",
        llama_cpp_dir / "quantize",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"llama.cpp quantize binary not found under {llama_cpp_dir}. "
        "Build llama.cpp first (see README)."
    )


def run(config_path="config/finetune_config.yaml"):
    """Merge the adapter and export a quantized GGUF using the config.

    Args:
        config_path: Path to the fine-tuning config YAML.

    Returns:
        The quantized GGUF path.
    """
    config = load_config(config_path)
    base_cfg = config["base_model"]
    train_cfg = config["training"]
    gguf_cfg = config["gguf"]

    merged_dir = merge_adapter(
        base_model_id=base_cfg["model_id"],
        adapter_dir=train_cfg["output_dir"],
        merged_dir=gguf_cfg["merged_dir"],
        trust_remote_code=base_cfg.get("trust_remote_code", False),
    )
    return export_gguf(
        merged_dir=merged_dir,
        output_path=gguf_cfg["output_path"],
        quant_type=gguf_cfg.get("quant_type", "Q4_K_M"),
        llama_cpp_dir=gguf_cfg.get("llama_cpp_dir", "third_party/llama.cpp"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge QLoRA adapter and export quantized GGUF. Author: A Taylor"
    )
    parser.add_argument("--config", type=str, default="config/finetune_config.yaml")
    args = parser.parse_args()
    run(args.config)
