#!/usr/bin/env python3
from nltk.parse.malt import MaltParser


def main():
    malt_parser = MaltParser("./maltparser", "engmalt.linear-1.7.mco")

    sentences = [
        "Flying planes can be dangerous.",
        "The parents of the bride and the groom were flying.",
        "The groom loves dangerous planes more than the bride.",
    ]

    for sentence in sentences:
        parse_trees = malt_parser.parse_all(sentence.split())
        print(f"\nSentence: {sentence}")

        for parse_tree in parse_trees:
            parse_tree.tree().pretty_print()


if __name__ == "__main__":
    main()
