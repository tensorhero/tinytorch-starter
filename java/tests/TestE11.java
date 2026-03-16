// TestE11.java — E11 Tokenization test driver
// Provided by tinytorch-starter. Do NOT modify.

import dev.tensorhero.tinytorch.Tokenizer;
import dev.tensorhero.tinytorch.Tokenizer.CharTokenizer;
import dev.tensorhero.tinytorch.Tokenizer.BPETokenizer;

public class TestE11 {
    public static void main(String[] args) {
        // =============================================================
        // Test 1–2: CharTokenizer — buildVocab + vocabSize
        // "hello" has unique chars: e, h, l, o → 4
        // =============================================================
        {
            CharTokenizer ct = new CharTokenizer();
            ct.buildVocab("hello");
            emit("char_vocab_size", String.valueOf(ct.vocabSize()));
        }

        // =============================================================
        // Test 3: CharTokenizer — encode
        // sorted: e=0, h=1, l=2, o=3
        // "hello" → [1, 0, 2, 2, 3]
        // =============================================================
        {
            CharTokenizer ct = new CharTokenizer();
            ct.buildVocab("hello");
            emit("char_encode", intArrayStr(ct.encode("hello")));
        }

        // =============================================================
        // Test 4: CharTokenizer — decode
        // [1, 0, 2, 2, 3] → "hello"
        // =============================================================
        {
            CharTokenizer ct = new CharTokenizer();
            ct.buildVocab("hello");
            emit("char_decode", ct.decode(new int[]{1, 0, 2, 2, 3}));
        }

        // =============================================================
        // Test 5: CharTokenizer — roundtrip
        // decode(encode(text)) == text
        // =============================================================
        {
            CharTokenizer ct = new CharTokenizer();
            String text = "the quick brown fox";
            ct.buildVocab(text);
            emit("char_roundtrip", ct.decode(ct.encode(text)));
        }

        // =============================================================
        // Test 6: CharTokenizer — vocab includes space
        // "a b" has chars: ' ', a, b → 3
        // =============================================================
        {
            CharTokenizer ct = new CharTokenizer();
            ct.buildVocab("a b");
            emit("char_space_vocab_size", String.valueOf(ct.vocabSize()));
            emit("char_space_encode", intArrayStr(ct.encode("a b")));
        }

        // =============================================================
        // Test 7: CharTokenizer — encode length
        // =============================================================
        {
            CharTokenizer ct = new CharTokenizer();
            ct.buildVocab("abcabc");
            emit("char_encode_length", String.valueOf(ct.encode("abcabc").length));
        }

        // =============================================================
        // Test 8–9: BPETokenizer — train + vocabSize
        // "aaabdaaabac" with vocabSize=6
        // chars: a,b,c,d → 4 base, +2 merges → 6
        // =============================================================
        {
            BPETokenizer bpe = new BPETokenizer();
            bpe.train("aaabdaaabac", 6);
            emit("bpe_vocab_size", String.valueOf(bpe.vocabSize()));
        }

        // =============================================================
        // Test 10: BPETokenizer — encode
        // After training on "aaabdaaabac" with vocabSize=6:
        //   merges: (a,a)→aa(4), (aa,a)→aaa(5)
        //   encode: [aaa,b,d,aaa,b,a,c] → [5,1,3,5,1,0,2]
        // =============================================================
        {
            BPETokenizer bpe = new BPETokenizer();
            bpe.train("aaabdaaabac", 6);
            emit("bpe_encode", intArrayStr(bpe.encode("aaabdaaabac")));
        }

        // =============================================================
        // Test 11: BPETokenizer — decode
        // [5,1,3,5,1,0,2] → "aaabdaaabac"
        // =============================================================
        {
            BPETokenizer bpe = new BPETokenizer();
            bpe.train("aaabdaaabac", 6);
            emit("bpe_decode", bpe.decode(new int[]{5, 1, 3, 5, 1, 0, 2}));
        }

        // =============================================================
        // Test 12: BPETokenizer — roundtrip
        // =============================================================
        {
            BPETokenizer bpe = new BPETokenizer();
            String text = "aaabdaaabac";
            bpe.train(text, 6);
            emit("bpe_roundtrip", bpe.decode(bpe.encode(text)));
        }

        // =============================================================
        // Test 13: BPETokenizer — compression
        // BPE tokens should be shorter than char tokens
        // =============================================================
        {
            String text = "aaabdaaabac";
            CharTokenizer ct = new CharTokenizer();
            ct.buildVocab(text);
            int charLen = ct.encode(text).length;

            BPETokenizer bpe = new BPETokenizer();
            bpe.train(text, 6);
            int bpeLen = bpe.encode(text).length;

            emit("bpe_compression", String.valueOf(bpeLen < charLen));
        }

        // =============================================================
        // Test 14: BPETokenizer — no merge (vocabSize = base chars)
        // vocabSize = 4 → no merges, behaves like CharTokenizer
        // =============================================================
        {
            BPETokenizer bpe = new BPETokenizer();
            bpe.train("aaabdaaabac", 4);
            emit("bpe_no_merge_size", String.valueOf(bpe.vocabSize()));
            emit("bpe_no_merge_len", String.valueOf(bpe.encode("aaabdaaabac").length));
        }

        // =============================================================
        // Test 15: BPETokenizer — longer text roundtrip
        // =============================================================
        {
            BPETokenizer bpe = new BPETokenizer();
            String text = "abababababcdcdcdcd";
            bpe.train(text, 8);
            emit("bpe_long_roundtrip", bpe.decode(bpe.encode(text)));
            emit("bpe_long_vocab_size", String.valueOf(bpe.vocabSize()));
        }

        // =============================================================
        // Test 16: BPETokenizer — encode on unseen ordering
        // Train on "aabb", encode "abab"
        // =============================================================
        {
            BPETokenizer bpe = new BPETokenizer();
            bpe.train("aabb", 4);
            emit("bpe_unseen_roundtrip", bpe.decode(bpe.encode("abab")));
        }
    }

    static void emit(String testName, String result) {
        System.out.println("TEST:" + testName);
        System.out.println("RESULT:" + result);
    }

    static String intArrayStr(int[] arr) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < arr.length; i++) {
            if (i > 0) sb.append(",");
            sb.append(arr[i]);
        }
        sb.append("]");
        return sb.toString();
    }
}
