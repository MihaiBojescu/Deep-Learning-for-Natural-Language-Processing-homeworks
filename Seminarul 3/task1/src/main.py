#!/usr/bin/env python3

from collections import defaultdict
from transformers import AutoTokenizer, PreTrainedTokenizer


def main():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    corpus = ["there is a big house", "i buy a house", "they buy the new house"]

    word_freqs = compute_frequencies(tokenizer, corpus)
    alphabet = compute_alphabet(word_freqs)
    augmented_alphabet = augment_alphabet(alphabet)

    splits = compute_splits(word_freqs)
    merges, vocab = merge_all(word_freqs, augmented_alphabet, splits)
    tokens = tokenize(tokenizer, merges, " house prices are high")

    print(tokens)


def compute_frequencies(tokenizer: PreTrainedTokenizer, corpus: list[str]):
    word_freqs = defaultdict(int)

    for text in corpus:
        words_with_offsets = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(
            text
        )
        new_words = [word for word, offset in words_with_offsets]
        for word in new_words:
            word_freqs[word] += 1

    return word_freqs


def compute_alphabet(word_freqs: defaultdict[any, int]):
    alphabet = []

    for word in word_freqs.keys():
        for letter in word:
            if letter not in alphabet:
                alphabet.append(letter)
    alphabet.sort()

    return alphabet


def augment_alphabet(alphabet: list[str]):
    augmented_alphabet = ["<|endoftext|>"] + alphabet.copy()
    return augmented_alphabet


def compute_splits(word_freqs: defaultdict[any, int]):
    splits = {word: list(word) for word in word_freqs.keys()}
    return splits


def compute_pair_freqs(word_freqs: defaultdict[any, int], splits: dict[str, list[str]]):
    pair_freqs = defaultdict(int)

    for word, freq in word_freqs.items():
        split = splits[word]

        if len(split) == 1:
            continue
        for i in range(len(split) - 1):
            pair = (split[i], split[i + 1])
            pair_freqs[pair] += freq

    return pair_freqs


def compute_best_pair(pair_freqs: defaultdict[any, int]):
    best_pair = ""
    max_freq: int = -1

    for pair, freq in pair_freqs.items():
        if max_freq is None or max_freq < freq:
            best_pair = pair
            max_freq = freq

    return best_pair, max_freq


def merge_all(word_freqs: defaultdict[any, int], vocabulary: list[str], splits):
    merges: dict[str, str] = {}
    vocab = vocabulary.copy()
    vocab_size = 60

    while len(vocab) < vocab_size:
        pair_freqs = compute_pair_freqs(word_freqs, splits)
        best_pair, max_freq = compute_best_pair(pair_freqs)

        if len(best_pair) == 0:
            break

        splits = merge_pair(*best_pair, splits, word_freqs)
        merges[best_pair] = best_pair[0] + best_pair[1]
        vocab.append(best_pair[0] + best_pair[1])

    return merges, vocab


def merge_pair(
    a: str, b: str, splits: dict[str, list[str]], word_freqs: defaultdict[any, int]
):
    for word in word_freqs:
        split = splits[word]
        if len(split) == 1:
            continue

        i = 0
        while i < len(split) - 1:
            if split[i] == a and split[i + 1] == b:
                split = split[:i] + [a + b] + split[i + 2 :]
            else:
                i += 1
        splits[word] = split
    return splits


def tokenize(tokenizer: PreTrainedTokenizer, merges: dict[str, str], text: str):
    pre_tokenize_result = tokenizer._tokenizer.pre_tokenizer.pre_tokenize_str(text)
    pre_tokenized_text = [word for word, offset in pre_tokenize_result]
    splits = [[l for l in word] for word in pre_tokenized_text]
    for pair, merge in merges.items():
        for idx, split in enumerate(splits):
            i = 0
            while i < len(split) - 1:
                if split[i] == pair[0] and split[i + 1] == pair[1]:
                    split = split[:i] + [merge] + split[i + 2 :]
                else:
                    i += 1
            splits[idx] = split

    return sum(splits, [])


if __name__ == "__main__":
    main()
