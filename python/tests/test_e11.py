"""test_e11.py — E11 Tokenization test driver.

Provided by tinytorch-starter. Do NOT modify.
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from tinytorch.tokenizer import CharTokenizer, BPETokenizer


def emit(test_name: str, result: str) -> None:
    print(f"TEST:{test_name}")
    print(f"RESULT:{result}")


def int_list_str(arr: list[int]) -> str:
    return "[" + ",".join(str(x) for x in arr) + "]"


def main() -> None:
    # =============================================================
    # Test 1–2: CharTokenizer — build_vocab + vocab_size
    # "hello" has unique chars: e, h, l, o → 4
    # =============================================================
    ct = CharTokenizer()
    ct.build_vocab("hello")
    emit("char_vocab_size", str(ct.vocab_size()))

    # =============================================================
    # Test 3: CharTokenizer — encode
    # sorted: e=0, h=1, l=2, o=3
    # "hello" → [1, 0, 2, 2, 3]
    # =============================================================
    ct = CharTokenizer()
    ct.build_vocab("hello")
    emit("char_encode", int_list_str(ct.encode("hello")))

    # =============================================================
    # Test 4: CharTokenizer — decode
    # [1, 0, 2, 2, 3] → "hello"
    # =============================================================
    ct = CharTokenizer()
    ct.build_vocab("hello")
    emit("char_decode", ct.decode([1, 0, 2, 2, 3]))

    # =============================================================
    # Test 5: CharTokenizer — roundtrip
    # decode(encode(text)) == text
    # =============================================================
    ct = CharTokenizer()
    text = "the quick brown fox"
    ct.build_vocab(text)
    emit("char_roundtrip", ct.decode(ct.encode(text)))

    # =============================================================
    # Test 6: CharTokenizer — vocab includes space
    # "a b" has chars: ' ', a, b → 3
    # =============================================================
    ct = CharTokenizer()
    ct.build_vocab("a b")
    emit("char_space_vocab_size", str(ct.vocab_size()))
    emit("char_space_encode", int_list_str(ct.encode("a b")))

    # =============================================================
    # Test 7: CharTokenizer — encode length
    # =============================================================
    ct = CharTokenizer()
    ct.build_vocab("abcabc")
    emit("char_encode_length", str(len(ct.encode("abcabc"))))

    # =============================================================
    # Test 8–9: BPETokenizer — train + vocab_size
    # "aaabdaaabac" with vocab_size=6
    # chars: a,b,c,d → 4 base, +2 merges → 6
    # =============================================================
    bpe = BPETokenizer()
    bpe.train("aaabdaaabac", 6)
    emit("bpe_vocab_size", str(bpe.vocab_size()))

    # =============================================================
    # Test 10: BPETokenizer — encode
    # After training on "aaabdaaabac" with vocab_size=6:
    #   merges: (a,a)→aa(4), (aa,a)→aaa(5)
    #   encode: [aaa,b,d,aaa,b,a,c] → [5,1,3,5,1,0,2]
    # =============================================================
    bpe = BPETokenizer()
    bpe.train("aaabdaaabac", 6)
    emit("bpe_encode", int_list_str(bpe.encode("aaabdaaabac")))

    # =============================================================
    # Test 11: BPETokenizer — decode
    # [5,1,3,5,1,0,2] → "aaabdaaabac"
    # =============================================================
    bpe = BPETokenizer()
    bpe.train("aaabdaaabac", 6)
    emit("bpe_decode", bpe.decode([5, 1, 3, 5, 1, 0, 2]))

    # =============================================================
    # Test 12: BPETokenizer — roundtrip
    # =============================================================
    bpe = BPETokenizer()
    text = "aaabdaaabac"
    bpe.train(text, 6)
    emit("bpe_roundtrip", bpe.decode(bpe.encode(text)))

    # =============================================================
    # Test 13: BPETokenizer — compression
    # BPE tokens should be shorter than char tokens
    # =============================================================
    text = "aaabdaaabac"
    ct = CharTokenizer()
    ct.build_vocab(text)
    char_len = len(ct.encode(text))

    bpe = BPETokenizer()
    bpe.train(text, 6)
    bpe_len = len(bpe.encode(text))

    emit("bpe_compression", str(bpe_len < char_len).lower())

    # =============================================================
    # Test 14: BPETokenizer — no merge (vocab_size = base chars)
    # vocab_size = 4 → no merges, behaves like CharTokenizer
    # =============================================================
    bpe = BPETokenizer()
    bpe.train("aaabdaaabac", 4)
    emit("bpe_no_merge_size", str(bpe.vocab_size()))
    emit("bpe_no_merge_len", str(len(bpe.encode("aaabdaaabac"))))

    # =============================================================
    # Test 15: BPETokenizer — longer text roundtrip
    # =============================================================
    bpe = BPETokenizer()
    text = "abababababcdcdcdcd"
    bpe.train(text, 8)
    emit("bpe_long_roundtrip", bpe.decode(bpe.encode(text)))
    emit("bpe_long_vocab_size", str(bpe.vocab_size()))

    # =============================================================
    # Test 16: BPETokenizer — encode on unseen ordering
    # Train on "aabb", encode "abab"
    # =============================================================
    bpe = BPETokenizer()
    bpe.train("aabb", 4)
    emit("bpe_unseen_roundtrip", bpe.decode(bpe.encode("abab")))


if __name__ == "__main__":
    main()
