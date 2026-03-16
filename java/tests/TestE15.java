// TestE15.java — E15 GPT & Generate test driver
// Provided by tinytorch-starter. Do NOT modify.

import dev.tensorhero.tinytorch.GPT;
import dev.tensorhero.tinytorch.Tensor;
import dev.tensorhero.tinytorch.Tokenizer;
import dev.tensorhero.tinynum.NDArray;

public class TestE15 {
    public static void main(String[] args) {
        // ============================================================
        // Setup: build a small GPT
        // ============================================================

        int vocabSize = 12;
        int dim = 16;
        int numHeads = 2;
        int numLayers = 2;
        int maxLen = 64;

        GPT gpt = new GPT(vocabSize, dim, numHeads, numLayers, maxLen);

        // ============================================================
        // Part 1: Forward pass
        // ============================================================

        int[] tokenIds = {0, 3, 7, 1};
        Tensor logits = gpt.forward(tokenIds);

        // 1. Output shape: [seqLen, vocabSize] = [4, 12]
        emit("forward_shape", shapeStr(logits.shape()));

        // 2. Output ndim = 2
        emit("forward_ndim", String.valueOf(logits.ndim()));

        // 3. Forward with single token
        int[] singleToken = {5};
        Tensor singleLogits = gpt.forward(singleToken);
        emit("forward_single_shape", shapeStr(singleLogits.shape()));

        // ============================================================
        // Part 2: Parameters and children
        // ============================================================

        // 4. Parameters count > 0
        int paramCount = gpt.parameters().size();
        emit("params_positive", String.valueOf(paramCount > 0));

        // 5. Children count = numLayers (each transformer block)
        emit("children_count", String.valueOf(gpt.children().size()));

        // 6. Specific parameter count calculation:
        //    tokenEmb: 1 (weight)
        //    posEnc: 0 (sinusoidal)
        //    per block: 16 (ln1:2 + attn:8 + ln2:2 + mlp:4)
        //    lnFinal: 2
        //    lmHead: 2 (weight + bias)
        //    total = 1 + 0 + 2*16 + 2 + 2 = 37
        emit("params_count", String.valueOf(paramCount));

        // ============================================================
        // Part 3: Generate
        // ============================================================

        // Build a simple char tokenizer
        Tokenizer.CharTokenizer tok = new Tokenizer.CharTokenizer();
        tok.buildVocab("abcdefghijkl");  // 12 chars = vocabSize

        // 7. Generate with temperature=0 (greedy, deterministic)
        String gen1 = gpt.generate(tok, "abc", 5, 0.0f);
        // Length should be prompt(3) + generated(5) = 8
        emit("generate_length", String.valueOf(gen1.length()));

        // 8. Greedy is deterministic: two calls produce same output
        String gen2 = gpt.generate(tok, "abc", 5, 0.0f);
        emit("generate_deterministic", String.valueOf(gen1.equals(gen2)).toLowerCase());

        // 9. Generated text starts with prompt
        emit("generate_starts_with_prompt", String.valueOf(gen1.startsWith("abc")).toLowerCase());

        // 10. Generate with temperature > 0
        String gen3 = gpt.generate(tok, "ab", 3, 1.0f);
        emit("generate_temp_length", String.valueOf(gen3.length()));

        // 11. Generate with empty-like prompt (single char)
        String gen4 = gpt.generate(tok, "a", 4, 0.0f);
        emit("generate_short_prompt_length", String.valueOf(gen4.length()));

        // 12. All generated characters are within vocabulary
        boolean allInVocab = true;
        for (char c : gen1.toCharArray()) {
            if (tok.charToId.get(c) == null) {
                allInVocab = false;
                break;
            }
        }
        emit("generate_valid_vocab", String.valueOf(allInVocab).toLowerCase());
    }

    static void emit(String testName, String result) {
        System.out.println("TEST:" + testName);
        System.out.println("RESULT:" + result);
    }

    static String shapeStr(int[] shape) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < shape.length; i++) {
            if (i > 0) sb.append(",");
            sb.append(shape[i]);
        }
        return sb.toString();
    }

    static String floatStr(float value) {
        return String.format("%.6f", value);
    }
}
