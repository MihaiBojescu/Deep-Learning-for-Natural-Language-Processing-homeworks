#!/usr/bin/env python3

from nltk.corpus import wordnet
from random import choice


def main():
    word = pick_word()
    print(f'Word is "{word}"')
    guess_word(word)


def pick_word():
    words: list[str] = []

    with open("../static/wordlist.txt", "r", encoding="utf-8") as file:
        for line in file.readlines():
            words.append(line.strip().lower())

    word = choice(words)

    return word


def guess_word(word: str):
    player_word = input("Enter a related word: ").strip().lower()
    similarity_score = get_similarity_score(word, player_word)

    print(
        f'Similarity score between "{word}" and "{player_word}" is {similarity_score:.2f}'
    )

    if similarity_score > 0.8:
        print("Very close match!")
        return

    if similarity_score > 0.5:
        print("Close match!")
        return

    if similarity_score > 0.3:
        print("Quite far off!")
        return

    print("Really far off!")


def get_similarity_score(word: str, player_word: str):
    word_synsets = wordnet.synsets(word)
    player_word_synsets = wordnet.synsets(player_word)
    max_similarity_score = 0

    for word_synset in word_synsets:
        for player_word_synset in player_word_synsets:
            similarity_score = word_synset.wup_similarity(player_word_synset)

            if similarity_score and similarity_score > max_similarity_score:
                max_similarity_score = similarity_score

    return max_similarity_score


if __name__ == "__main__":
    main()
