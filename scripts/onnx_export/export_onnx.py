"""Export MOSAIC GPT-2 checkpoint to ONNX format for browser inference.

This script converts a PyTorch Lightning checkpoint into:
1. A HuggingFace-compatible GPT-2 model
2. An ONNX model (optionally quantized to int8)
3. A tokenizer_config.json with HDTC metadata for the JS tokenizer

Usage:
    python scripts/export_onnx.py \
        --checkpoint path/to/checkpoint.ckpt \
        --output_dir exports/hdtc_coconut \
        --quantize

Requirements:
    pip install optimum[onnxruntime] onnxruntime
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path

import torch
from transformers import GPT2Config, GPT2LMHeadModel


def extract_gpt2_state_dict(checkpoint_path: str) -> tuple[OrderedDict, dict]:
    """Extract GPT-2 weights from Lightning checkpoint.

    Lightning wraps the model as:
        model.model.transformer.wte.weight -> transformer.wte.weight

    Args:
        checkpoint_path: Path to .ckpt file.

    Returns:
        Tuple of (cleaned state_dict, metadata dict with vocab_size etc.)
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["state_dict"]

    # Extract model metadata from weight shapes
    wte_key = "model.model.transformer.wte.weight"
    wpe_key = "model.model.transformer.wpe.weight"

    if wte_key not in state_dict:
        raise ValueError(f"Cannot find {wte_key} in checkpoint state_dict")

    vocab_size = state_dict[wte_key].shape[0]
    n_embd = state_dict[wte_key].shape[1]
    n_positions = state_dict[wpe_key].shape[0] if wpe_key in state_dict else 2048

    # Count layers by looking for layer norm weights
    n_layer = 0
    while f"model.model.transformer.h.{n_layer}.ln_1.weight" in state_dict:
        n_layer += 1

    # Get n_head from attention weight shape
    attn_key = f"model.model.transformer.h.0.attn.c_attn.weight"
    if attn_key in state_dict:
        # c_attn projects to 3 * n_embd (Q, K, V), so shape is [n_embd, 3*n_embd]
        n_head = n_embd // (state_dict[attn_key].shape[-1] // 3)
        # GPT-2 typically: n_head = n_embd // head_dim, and c_attn maps n_embd -> 3*n_embd
        # For xs config: n_embd=384, n_head=12, head_dim=32
        # Detect from config: try common head counts
        for candidate_n_head in [12, 8, 6, 4, 16]:
            if n_embd % candidate_n_head == 0:
                n_head = candidate_n_head
                break
    else:
        n_head = 12  # Default for xs config

    # Try to get n_head from hyperparameters if available
    if "hyper_parameters" in checkpoint:
        hp = checkpoint["hyper_parameters"]
        model_name = hp.get("model_name", "gpt2-xs")
        size_map = {
            "xxs": 4, "xs": 12, "s": 12, "m": 16,
        }
        size = model_name.split("-")[-1] if "-" in model_name else "xs"
        n_head = size_map.get(size, 12)

    # Strip "model.model." prefix from keys
    prefix = "model.model."
    cleaned = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith(prefix):
            new_key = key[len(prefix):]
            cleaned[new_key] = value

    metadata = {
        "vocab_size": vocab_size,
        "n_embd": n_embd,
        "n_positions": n_positions,
        "n_layer": n_layer,
        "n_head": n_head,
    }

    return cleaned, metadata


def create_hf_model(state_dict: OrderedDict, metadata: dict) -> GPT2LMHeadModel:
    """Create a HuggingFace GPT2LMHeadModel from extracted weights.

    Args:
        state_dict: Cleaned state dict with HF-compatible keys.
        metadata: Dict with vocab_size, n_embd, n_positions, n_layer, n_head.

    Returns:
        Initialized GPT2LMHeadModel.
    """
    config = GPT2Config(
        vocab_size=metadata["vocab_size"],
        n_embd=metadata["n_embd"],
        n_layer=metadata["n_layer"],
        n_head=metadata["n_head"],
        n_positions=metadata["n_positions"],
        bos_token_id=0,  # SOS
        eos_token_id=1,  # EOS
        pad_token_id=2,  # PAD
    )

    model = GPT2LMHeadModel(config)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    return model


def export_tokenizer_config(output_dir: Path, metadata: dict) -> None:
    """Export HDTC tokenizer configuration for the JavaScript tokenizer.

    Args:
        output_dir: Directory to write tokenizer_config.json.
        metadata: Model metadata dict.
    """
    # HDTC token constants
    config = {
        "tokenizer_type": "hdtc",
        "special_tokens": {
            "SOS": 0,
            "EOS": 1,
            "PAD": 2,
            "COMM_START": 3,
            "COMM_END": 4,
            "LEDGE": 5,
            "REDGE": 6,
            "SUPER_START": 7,
            "SUPER_END": 8,
            "TYPE_RING": 9,
            "TYPE_FUNC": 10,
            "TYPE_SINGLETON": 11,
        },
        "IDX_OFFSET": 12,
        "vocab_size": metadata["vocab_size"],
        "labeled_graph": True,
        "num_atom_types": 10,  # NUM_ATOM_TYPES (9 + 1 unknown)
        "num_bond_types": 5,   # NUM_BOND_TYPES (4 + 1 unknown)
        "atom_types": ["C", "N", "O", "F", "P", "S", "Cl", "Br", "I", "Unknown"],
        "bond_types": ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC", "Unknown"],
        "max_num_nodes": metadata["vocab_size"] - 12 - 10 - 5,  # vocab - offset - atoms - bonds
        "n_positions": metadata["n_positions"],
    }

    with open(output_dir / "tokenizer_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"  Saved tokenizer_config.json (max_num_nodes={config['max_num_nodes']})")


def verify_onnx_export(hf_model_dir: Path, onnx_dir: Path) -> None:
    """Verify ONNX model produces same outputs as PyTorch model.

    Args:
        hf_model_dir: Path to saved HuggingFace model.
        onnx_dir: Path to exported ONNX model directory.
    """
    try:
        import numpy as np
        import onnxruntime as ort

        # Load PyTorch model
        pt_model = GPT2LMHeadModel.from_pretrained(str(hf_model_dir))
        pt_model.eval()

        # Load ONNX model
        onnx_path = onnx_dir / "model.onnx"
        if not onnx_path.exists():
            # Check for decoder_model.onnx (optimum naming convention)
            onnx_path = onnx_dir / "decoder_model.onnx"
        session = ort.InferenceSession(str(onnx_path))

        # Test with a sample input (SOS token)
        input_ids = torch.tensor([[0, 3, 9, 12, 15, 4]], dtype=torch.long)

        # PyTorch forward
        with torch.no_grad():
            pt_logits = pt_model(input_ids=input_ids).logits

        # ONNX forward
        ort_inputs = {"input_ids": input_ids.numpy()}
        ort_outputs = session.run(None, ort_inputs)
        onnx_logits = ort_outputs[0]

        # Compare
        max_diff = np.max(np.abs(pt_logits.numpy() - onnx_logits))
        print(f"  Verification: max logit difference = {max_diff:.6f}")
        if max_diff < 1e-3:
            print("  PASS: ONNX output matches PyTorch")
        else:
            print(f"  WARNING: Large difference ({max_diff:.4f}), may affect generation quality")

    except ImportError:
        print("  Skipping verification (onnxruntime not installed)")
    except Exception as e:
        print(f"  Verification failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Export MOSAIC model to ONNX")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to Lightning checkpoint (.ckpt)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="exports/hdtc_coconut",
        help="Output directory for exported model",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Quantize ONNX model to int8",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify ONNX output matches PyTorch",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    hf_dir = output_dir / "hf_model"
    onnx_dir = output_dir / "onnx"

    # Step 1: Extract weights and create HF model
    print("Step 1: Extracting GPT-2 weights from Lightning checkpoint...")
    state_dict, metadata = extract_gpt2_state_dict(args.checkpoint)
    print(f"  Model config: {metadata}")

    model = create_hf_model(state_dict, metadata)
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Step 2: Save as HuggingFace format
    print("Step 2: Saving HuggingFace model...")
    hf_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(hf_dir))
    print(f"  Saved to {hf_dir}")

    # Step 3: Export tokenizer config
    print("Step 3: Exporting tokenizer config...")
    output_dir.mkdir(parents=True, exist_ok=True)
    export_tokenizer_config(output_dir, metadata)

    # Step 4: Export to ONNX
    print("Step 4: Exporting to ONNX...")
    onnx_dir.mkdir(parents=True, exist_ok=True)

    try:
        from optimum.onnxruntime import ORTModelForCausalLM

        ort_model = ORTModelForCausalLM.from_pretrained(
            str(hf_dir), export=True
        )
        ort_model.save_pretrained(str(onnx_dir))
        print(f"  Saved ONNX model to {onnx_dir}")

    except ImportError:
        print("  optimum not installed, using torch.onnx.export fallback...")
        # Use a multi-token dummy input so GPT-2's causal attention mask
        # is traced correctly for variable-length sequences.
        # A single-token input bakes the mask as [1,1,1,1], breaking longer sequences.
        dummy_input = torch.tensor([[0, 3, 9, 12, 15, 4, 3, 10]], dtype=torch.long)
        onnx_path = onnx_dir / "model.onnx"

        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_path),
            input_names=["input_ids"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "logits": {0: "batch_size", 1: "sequence_length"},
            },
            opset_version=14,
        )
        print(f"  Saved ONNX model to {onnx_path}")

    # Step 5: Quantize if requested
    if args.quantize:
        print("Step 5: Quantizing to int8...")
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType

            onnx_path = onnx_dir / "model.onnx"
            if not onnx_path.exists():
                onnx_path = onnx_dir / "decoder_model.onnx"
            quantized_path = onnx_dir / "model_quantized.onnx"

            quantize_dynamic(
                str(onnx_path),
                str(quantized_path),
                weight_type=QuantType.QInt8,
            )

            # Report size reduction
            orig_size = onnx_path.stat().st_size / (1024 * 1024)
            quant_size = quantized_path.stat().st_size / (1024 * 1024)
            print(f"  Original: {orig_size:.1f} MB -> Quantized: {quant_size:.1f} MB")
            print(f"  Reduction: {(1 - quant_size/orig_size)*100:.1f}%")

        except ImportError:
            print("  Skipping quantization (onnxruntime.quantization not available)")

    # Step 6: Verify if requested
    if args.verify:
        print("Step 6: Verifying ONNX export...")
        verify_onnx_export(hf_dir, onnx_dir)

    # Copy config files to onnx dir for Transformers.js compatibility
    import shutil
    config_src = hf_dir / "config.json"
    if config_src.exists():
        shutil.copy(config_src, onnx_dir / "config.json")

    # Create generation_config.json
    gen_config = {
        "bos_token_id": 0,
        "eos_token_id": 1,
        "pad_token_id": 2,
        "max_length": metadata["n_positions"],
        "do_sample": True,
        "top_k": 10,
        "temperature": 1.0,
    }
    with open(onnx_dir / "generation_config.json", "w") as f:
        json.dump(gen_config, f, indent=2)

    print("\nExport complete!")
    print(f"  HuggingFace model: {hf_dir}")
    print(f"  ONNX model: {onnx_dir}")
    print(f"  Tokenizer config: {output_dir / 'tokenizer_config.json'}")


if __name__ == "__main__":
    main()
