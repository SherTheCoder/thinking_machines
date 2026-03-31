"""
quantize_awq.py
===============
Applies Activation-aware Weight Quantisation (AWQ) to the merged QLoRA model,
calibrating on the actual course dialogue dataset.

"""

import argparse
import json
import random
from pathlib import Path
from typing import Any

# AutoAWQ import
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL_PATH: str = str(
    Path(__file__).parent.parent / "checkpoints" / "merged_socratic"
)

DEFAULT_OUTPUT_PATH: str = str(
    Path(__file__).parent.parent / "checkpoints" / "socratic_awq_q4"
)

CALIBRATION_DATA_PATH: str = str(
    Path(__file__).parent.parent / "data" / "socratic_dialogues.jsonl"
)

# AWQ quantisation parameters
AWQ_CONFIG: dict[str, Any] = {
    "zero_point": True,      
    "q_group_size": 128,     
    "w_bit": 4,            
    "version": "GEMM",       
}

# Number of calibration samples
N_CALIBRATION_SAMPLES: int = 128
MAX_CALIBRATION_SEQ_LEN: int = 512


# ---------------------------------------------------------------------------
# Calibration Dataset Preparation
# ---------------------------------------------------------------------------

def load_calibration_texts(
    data_path: str,
    n_samples: int,
    max_seq_len: int,
    tokeniser: AutoTokenizer,
) -> list[list[int]]:
    """
    Loads and tokenises calibration samples from the course-specific JSONL dataset.

    Args:
        data_path: Path to the JSONL file produced by generate_mock_data.py.
        n_samples: Number of calibration sequences to extract.
        max_seq_len: Maximum token length per sequence (truncated if longer).
        tokeniser: Tokeniser matching the model being quantised.

    Returns:
        List of tokenised sequences (list of int token ids), each of length
        up to max_seq_len.
    """
    data_path_obj = Path(data_path)
    if not data_path_obj.exists():
        raise FileNotFoundError(
            f"Calibration data not found at {data_path}.\n"
            "Run ml_pipeline/generate_mock_data.py first:\n"
            "  python ml_pipeline/generate_mock_data.py"
        )

    records: list[dict[str, Any]] = []
    with open(data_path_obj, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if len(records) < n_samples:
        raise ValueError(
            f"Dataset contains only {len(records)} records, "
            f"but {n_samples} calibration samples were requested. "
            "Regenerate the dataset or reduce N_CALIBRATION_SAMPLES."
        )

    # Random sample for representativeness across all topic areas
    random.seed(42)
    sampled = random.sample(records, n_samples)

    tokenised: list[list[int]] = []
    for record in sampled:
        text: str = record["text"]
        token_ids: list[int] = tokeniser.encode(
            text,
            truncation=True,
            max_length=max_seq_len,
            add_special_tokens=True,
        )
        tokenised.append(token_ids)

    print(
        f"[quantize_awq] Loaded {len(tokenised)} calibration sequences "
        f"from {data_path} (max_seq_len={max_seq_len})"
    )
    return tokenised


# ---------------------------------------------------------------------------
# AWQ Quantisation
# ---------------------------------------------------------------------------

def quantize(model_path: str, output_path: str) -> None:
    """
    Applies AWQ quantisation to the merged model and saves the result.

    Args:
        model_path: Path to the merged full-precision Llama-3-8B model.
        output_path: Directory where the AWQ-quantised model will be saved.
    """
    print(f"[quantize_awq] Loading model from: {model_path}")
    print(f"[quantize_awq] Output path: {output_path}")
    print(f"[quantize_awq] AWQ config: {AWQ_CONFIG}")

    # Load tokeniser
    tokeniser = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    tokeniser.pad_token = tokeniser.eos_token

    # device_map="cuda:0" pins to the A100 GPU for calibration forward passes
    model = AutoAWQForCausalLM.from_pretrained(
        model_path,
        device_map="cuda:0",
        safetensors=True,
    )

    # -----------------------------------------------------------------------
    # Load calibration data from COURSE-SPECIFIC dataset
    # -----------------------------------------------------------------------
    calibration_data = load_calibration_texts(
        data_path=CALIBRATION_DATA_PATH,
        n_samples=N_CALIBRATION_SAMPLES,
        max_seq_len=MAX_CALIBRATION_SEQ_LEN,
        tokeniser=tokeniser,
    )

    print(
        f"[quantize_awq] Running AWQ calibration with {len(calibration_data)} "
        f"course-specific sequences..."
    )

    # Run AWQ quantisation with calibration
    model.quantize(
        tokenizer=tokeniser,
        quant_config=AWQ_CONFIG,
        calib_data=calibration_data,
    )

    # Save quantised model and tokeniser
    output_path_obj = Path(output_path)
    output_path_obj.mkdir(parents=True, exist_ok=True)

    model.save_quantized(str(output_path_obj))
    tokeniser.save_pretrained(str(output_path_obj))

    print(f"[quantize_awq] Quantisation complete. Model saved to: {output_path_obj}")
    print(
        "[quantize_awq] Memory footprint: ~4GB (vs. ~16GB BF16 / ~8GB NF4). "
        "Ready for vLLM deployment with --quantization awq."
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="AWQ quantisation for the Socratic AI tutor model"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Path to the merged full-precision model directory",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=DEFAULT_OUTPUT_PATH,
        help="Directory where the AWQ-quantised model will be saved",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    quantize(model_path=args.model_path, output_path=args.output_path)
