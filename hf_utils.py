"""Shared helpers for the experiments on pretrained Hugging Face models (Experiments A and C)."""

import torch
import torch.nn as nn

MLP_TYPES = ("gate_proj", "up_proj", "down_proj")   # names used by Llama/Qwen/SmolLM/OLMo-style models


def load_model(name, dtype=torch.bfloat16, device="cuda"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to(device).eval()   # transformers >= 4.56
    return model, tok


def mlp_linears(model, proj_type):
    """All nn.Linear modules called e.g. '...mlp.gate_proj', in layer order, as (full_name, module)."""
    found = [(n, m) for n, m in model.named_modules()
             if n.endswith("mlp." + proj_type) and isinstance(m, nn.Linear)]
    assert found, f"no mlp.{proj_type} layers found; is this a GLU model with Llama-style names?"
    return found


def eval_text(dataset, split_hint="test"):
    """Returns one long evaluation string. wikitext2: the standard test split. c4: the usual validation file."""
    from datasets import load_dataset
    if dataset == "wikitext2":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split_hint)
        return "\n\n".join(ds["text"])
    if dataset == "c4":
        ds = load_dataset("allenai/c4", data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
                          split="validation")
        return " ".join(ds[:1100]["text"])
    raise ValueError(dataset)


def tokenize_blocks(tok, text, seq_len, max_blocks=None):
    """Tokenize one long string and cut it into non-overlapping [n_blocks, seq_len] windows."""
    ids = tok(text, return_tensors="pt").input_ids[0]
    n = ids.numel() // seq_len
    if max_blocks:
        n = min(n, max_blocks)
    return ids[: n * seq_len].view(n, seq_len)


@torch.no_grad()
def perplexity(model, blocks, batch_size=4):
    """exp(mean next-token negative log-likelihood) over all windows."""
    device = next(model.parameters()).device
    nll, count = 0.0, 0
    for i in range(0, blocks.size(0), batch_size):
        x = blocks[i:i + batch_size].to(device)
        logits = model(x).logits[:, :-1].float()
        loss = nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), x[:, 1:].reshape(-1), reduction="sum")
        nll += loss.item()
        count += x[:, 1:].numel()
    return float(torch.exp(torch.tensor(nll / count)))
