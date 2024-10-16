#!/usr/bin/env python3
from nltk import CFG, ChartParser

grammar = CFG.fromstring(
    """
    S -> VP NP VP | NP VP

    NP -> DET N | ADJ N | N | NP CONJ NP | NP CP | NP PP
    CP -> COMP | COMP PP
    PP -> P NP
    VP -> V | V NP | AUX BE ADJ | BE VP
    
    BE -> "are" | "is" | "were" | "was" | "be"
    AUX -> "can"
    DET -> "the" | "The"
    N -> "planes" | "parents" | "bride" | "groom"
    V -> "loves" | "flying" | "Flying"
    ADJ -> "flying" | "dangerous" | "Flying"
    P -> "of" | "than"
    COMP -> "more"
    CONJ -> "and"
    """
)

sentences = [
    "Flying planes can be dangerous".split(),
    "The parents of the bride and the groom were flying".split(),
    "The groom loves dangerous planes more than the bride".split(),
]

cp = ChartParser(grammar)

for sentence in sentences:
    print(f"\nSentence: {' '.join(sentence)}")
    trees = list(cp.parse_all(sentence))

    for i, tree in enumerate(trees, 1):
        print(f"\nParse Tree {i}:")
        print(tree)

    if len(trees) != 2:
        print(f"Warning: Generated {len(trees)} trees instead of 2 for this sentence.")
