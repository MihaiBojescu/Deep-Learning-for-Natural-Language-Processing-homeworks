#!/usr/bin/env python3
import spacy


def main():
    nlp = spacy.load("en_core_web_sm")

    sentences = [
        "Flying planes can be dangerous.",
        "The parents of the bride and the groom were flying.",
        "The groom loves dangerous planes more than the bride.",
    ]

    for sentence in sentences:
        doc = nlp(sentence)
        print(f"\nSentence: {sentence}")
        for token in doc:
            print(
                f"{token.text:<12} {token.dep_:<12} {token.head.text:<12} {token.head.pos_:<6}"
            )


if __name__ == "__main__":
    main()
