"""test_e15.py — E15 GPT & Generate test driver.

Provided by tinytorch-starter. Do NOT modify.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from tinynum import NDArray
from tinytorch import Tensor
from tinytorch.gpt import GPT
from tinytorch.tokenizer import CharTokenizer


def emit(test_name: str, result: str) -> None:
    print(f"TEST:{test_name}")
    print(f"RESULT:{result}")


def shape_str(shape: tuple[int, ...]) -> str:
    return ",".join(str(s) for s in shape)


def float_str(value: float) -> str:
    return f"{value:.6f}"


def main() -> None:
    # ============================================================
    # Setup: build a small GPT
    # ============================================================

    vocab_size = 12
    dim = 16
    num_heads = 2
    num_layers = 2
    max_len = 64

    gpt = GPT(vocab_size, dim, num_heads, num_layers, max_len)

    # ============================================================
    # Part 1: Forward pass
    # ============================================================

    token_ids = [0, 3, 7, 1]
    logits = gpt.forward(token_ids)

    # 1. Output shape: [seqLen, vocabSize] = [4, 12]
    emit("forward_shape", shape_str(logits.data.get_shape()))

    # 2. Output ndim = 2
    emit("forward_ndim", str(len(logits.data.get_shape())))

    # 3. Forward with single token
    single_logits = gpt.forward([5])
    emit("forward_single_shape", shape_str(single_logits.data.get_shape()))

    # ============================================================
    # Part 2: Parameters and children
    # ============================================================

    # 4. Parameters count > 0
    param_count = len(gpt.parameters())
    emit("params_positive", str(param_count > 0).lower())

    # 5. Children count = numLayers (each transformer block)
    emit("children_count", str(len(gpt.children())))

    # 6. Specific parameter count
    emit("params_count", str(param_count))

    # ============================================================
    # Part 3: Generate
    # ============================================================

    # Build a simple char tokenizer
    tok = CharTokenizer()
    tok.build_vocab("abcdefghijkl")  # 12 chars = vocab_size

    # 7. Generate with temperature=0 (greedy, deterministic)
    gen1 = gpt.generate(tok, "abc", 5, 0.0)
    # Length should be prompt(3) + generated(5) = 8
    emit("generate_length", str(len(gen1)))

    # 8. Greedy is deterministic: two calls produce same output
    gen2 = gpt.generate(tok, "abc", 5, 0.0)
    emit("generate_deterministic", str(gen1 == gen2).lower())

    # 9. Generated text starts with prompt
    emit("generate_starts_with_prompt", str(gen1.startswith("abc")).lower())

    # 10. Generate with temperature > 0
    gen3 = gpt.generate(tok, "ab", 3, 1.0)
    emit("generate_temp_length", str(len(gen3)))

    # 11. Generate with single char prompt
    gen4 = gpt.generate(tok, "a", 4, 0.0)
    emit("generate_short_prompt_length", str(len(gen4)))

    # 12. All generated characters are within vocabulary
    all_in_vocab = all(c in tok.char_to_id for c in gen1)
    emit("generate_valid_vocab", str(all_in_vocab).lower())


if __name__ == "__main__":
    main()
