#!/usr/bin/env python3
from flask import Flask, request
from task2 import get_similarity_score, pick_word


app = Flask(__name__, static_url_path="", static_folder="../public")


@app.route("/api/words", methods=["GET"])
def get_word():
    word = pick_word()

    return {"word": word}


@app.route("/api/guesses", methods=["POST"])
def post_guess():
    content = request.json
    word = content["word"]
    user_word = content["userWord"]
    similarity = get_similarity_score(word, user_word)

    return {"similarity": similarity}


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
