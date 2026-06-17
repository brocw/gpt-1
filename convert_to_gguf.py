"""
Convert a model.pt checkpoint to GGUF format.

The output file can be quantized further with llama.cpp's quantize tool:
  ./llama-quantize model.gguf model-q4_k_m.gguf Q4_K_M

NOTE: llama.cpp inference of this file requires ROCm/HIP or CPU build of llama.cpp.
For quick Python inference use inference.py instead.
"""

import sys
import torch
import numpy as np
import gguf
from gpt import GPTConfig, GPT

ARCH = "gpt2"

def load_checkpoint(path: str):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]
    stoi = checkpoint["stoi"]
    itos = checkpoint["itos"]
    model = GPT(cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, cfg, stoi, itos


def build_tensor_map(model, cfg) -> dict[str, np.ndarray]:
    """Map PyTorch state_dict keys → GGUF tensor names."""
    sd = {k: v.float().numpy() for k, v in model.state_dict().items()}
    tensors = {}

    tensors["token_embd.weight"] = sd["token_embedding_table.weight"]
    tensors["position_embd.weight"] = sd["position_embedding_table.weight"]
    tensors["output_norm.weight"] = sd["ln_f.weight"]
    tensors["output_norm.bias"] = sd["ln_f.bias"]
    tensors["output.weight"] = sd["lm_head.weight"]

    for i in range(cfg.n_layer):
        prefix = f"blocks.{i}"
        out = f"blk.{i}"

        tensors[f"{out}.attn_norm.weight"] = sd[f"{prefix}.ln1.weight"]
        tensors[f"{out}.attn_norm.bias"]   = sd[f"{prefix}.ln1.bias"]

        # Combined QKV: shape [3*n_embd, n_embd] — matches llama.cpp gpt2 expectation
        tensors[f"{out}.attn_qkv.weight"]  = sd[f"{prefix}.sa.c_attn.weight"]
        tensors[f"{out}.attn_out.weight"]  = sd[f"{prefix}.sa.c_proj.weight"]

        tensors[f"{out}.ffn_norm.weight"]  = sd[f"{prefix}.ln2.weight"]
        tensors[f"{out}.ffn_norm.bias"]    = sd[f"{prefix}.ln2.bias"]

        # net.0 = first Linear (up), net.2 = second Linear (down)
        tensors[f"{out}.ffn_up.weight"]    = sd[f"{prefix}.ffwd.net.0.weight"]
        tensors[f"{out}.ffn_down.weight"]  = sd[f"{prefix}.ffwd.net.2.weight"]

    return tensors


def convert(checkpoint_path: str, output_path: str, use_f16: bool = False):
    print(f"Loading {checkpoint_path} ...")
    model, cfg, stoi, itos = load_checkpoint(checkpoint_path)

    writer = gguf.GGUFWriter(output_path, ARCH)

    # --- Model metadata ---
    writer.add_name("char-gpt")
    writer.add_block_count(cfg.n_layer)
    writer.add_context_length(cfg.block_size)
    writer.add_embedding_length(cfg.n_embd)
    writer.add_feed_forward_length(4 * cfg.n_embd)
    writer.add_head_count(cfg.n_head)
    writer.add_layer_norm_eps(1e-5)
    writer.add_vocab_size(cfg.vocab_size)
    writer.add_causal_attention(True)
    writer.add_file_type(1 if use_f16 else 0)  # 0=F32, 1=F16

    # --- Tokenizer ---
    # Character-level vocabulary stored as single-char tokens.
    # llama.cpp gpt2 tokenizer type; no BPE merges needed for char-level models.
    tokens = [itos[i] for i in range(cfg.vocab_size)]
    token_types = [gguf.TokenType.NORMAL] * cfg.vocab_size
    scores = [0.0] * cfg.vocab_size

    writer.add_tokenizer_model("gpt2")
    writer.add_token_list(tokens)
    writer.add_token_types(token_types)
    writer.add_token_scores(scores)
    writer.add_bos_token_id(stoi.get("\n", 0))
    writer.add_eos_token_id(stoi.get("\n", 0))

    # --- Tensors ---
    tensor_map = build_tensor_map(model, cfg)
    for name, arr in tensor_map.items():
        if use_f16:
            arr = arr.astype(np.float16)
        writer.add_tensor(name, arr)
        print(f"  {name:50s}  {arr.shape}")

    print(f"\nWriting {output_path} ...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print("Done.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="model.pt")
    parser.add_argument("--output",     default="model.gguf")
    parser.add_argument("--f16",        action="store_true",
                        help="store weights as float16 (half the file size, same quality)")
    args = parser.parse_args()
    convert(args.checkpoint, args.output, use_f16=args.f16)
