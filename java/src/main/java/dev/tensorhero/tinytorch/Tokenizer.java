package dev.tensorhero.tinytorch;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Tokenizers — convert text to integer token ID sequences.
 *
 * <p>Contains two tokenizer implementations:</p>
 * <ul>
 *   <li>{@link CharTokenizer} — one character = one token.</li>
 *   <li>{@link BPETokenizer} — Byte Pair Encoding, merges frequent pairs.</li>
 * </ul>
 */
public class Tokenizer {

    // ================================================================
    // CharTokenizer — character-level tokenization
    // ================================================================

    /**
     * Character-level tokenizer: each unique character maps to one token ID.
     * IDs are assigned in sorted character order.
     */
    public static class CharTokenizer {

        /** Character → token ID. */
        public Map<Character, Integer> charToId = new HashMap<>();

        /** Token ID → character. */
        public Map<Integer, Character> idToChar = new HashMap<>();

        // ================================================================
        // E11 — Tokenization
        // ================================================================

        /**
         * Builds vocabulary from text. Each unique character gets an ID
         * assigned in sorted character order (0, 1, 2, ...).
         *
         * @param text the training text
         */
        public void buildVocab(String text) {
            throw new UnsupportedOperationException("TODO: E11");
        }

        /**
         * Returns the vocabulary size.
         */
        public int vocabSize() {
            throw new UnsupportedOperationException("TODO: E11");
        }

        /**
         * Encodes text into token IDs.
         *
         * @param text the text to encode
         * @return array of token IDs
         */
        public int[] encode(String text) {
            throw new UnsupportedOperationException("TODO: E11");
        }

        /**
         * Decodes token IDs back to text.
         *
         * @param ids the token IDs to decode
         * @return the decoded text
         */
        public String decode(int[] ids) {
            throw new UnsupportedOperationException("TODO: E11");
        }
    }

    // ================================================================
    // BPETokenizer — Byte Pair Encoding
    // ================================================================

    /**
     * BPE tokenizer: starts from character-level vocabulary and iteratively
     * merges the most frequent adjacent token pairs until reaching the target
     * vocabulary size.
     */
    public static class BPETokenizer {

        /** Token string → token ID. */
        public Map<String, Integer> vocab = new HashMap<>();

        /** Token ID → token string. */
        public Map<Integer, String> reverseVocab = new HashMap<>();

        /** Ordered list of merge rules. Each entry is [left, right]. */
        public List<String[]> merges = new java.util.ArrayList<>();

        // ================================================================
        // E11 — Tokenization
        // ================================================================

        /**
         * Trains the BPE tokenizer on the given text.
         *
         * <p>Starts with a character-level vocabulary (sorted), then repeatedly
         * merges the most frequent adjacent token pair until vocabSize is reached.
         * Ties in frequency are broken by lexicographic order of the pair.</p>
         *
         * @param text      the training text
         * @param vocabSize the target vocabulary size
         */
        public void train(String text, int vocabSize) {
            throw new UnsupportedOperationException("TODO: E11");
        }

        /**
         * Returns the current vocabulary size.
         */
        public int vocabSize() {
            throw new UnsupportedOperationException("TODO: E11");
        }

        /**
         * Encodes text into token IDs by applying merge rules in training order.
         *
         * @param text the text to encode
         * @return array of token IDs
         */
        public int[] encode(String text) {
            throw new UnsupportedOperationException("TODO: E11");
        }

        /**
         * Decodes token IDs back to text.
         *
         * @param ids the token IDs to decode
         * @return the decoded text
         */
        public String decode(int[] ids) {
            throw new UnsupportedOperationException("TODO: E11");
        }
    }
}
