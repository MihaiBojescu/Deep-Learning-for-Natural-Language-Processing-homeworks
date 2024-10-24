#!/usr/bin/env python

import torch
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer
from googletrans import Translator

model = BertForQuestionAnswering.from_pretrained(
    "bert-large-uncased-whole-word-masking-finetuned-squad"
)
tokenizer = BertTokenizer.from_pretrained(
    "bert-large-uncased-whole-word-masking-finetuned-squad"
)


def question_answer(question: str, text: str) -> str:

    # tokenize question and text as a pair
    input_ids = tokenizer.encode(question, text)

    # string version of tokenized ids
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    # segment IDs
    # first occurence of [SEP] token
    sep_idx = input_ids.index(
        tokenizer.sep_token_id
    )  # number of tokens in segment A (question)
    num_seg_a = sep_idx + 1  # number of tokens in segment B (text)
    num_seg_b = len(input_ids) - num_seg_a

    # list of 0s and 1s for segment embeddings
    segment_ids = [0] * num_seg_a + [1] * num_seg_b
    assert len(segment_ids) == len(input_ids)

    # model output using input_ids and segment_ids
    output = model(
        torch.tensor([input_ids]), token_type_ids=torch.tensor([segment_ids])
    )

    # reconstructing the answer
    answer_start = torch.argmax(output.start_logits)
    answer_end = torch.argmax(output.end_logits)
    if answer_end >= answer_start:
        answer = tokens[answer_start]
        for i in range(answer_start + 1, answer_end + 1):
            if tokens[i][0:2] == "##":
                answer += tokens[i][2:]
            else:
                answer += " " + tokens[i]

    if answer.startswith("[CLS]"):
        answer = "Unable to find the answer to your question."

    return answer.capitalize()


text = """Great Britain (commonly shortened to Britain) is an island in the North Atlantic Ocean off the north-west coast of continental Europe, consisting of the countries England, Scotland and Wales. With an area of 209,331 km2 (80,823 sq mi), it is the largest of the British Isles, the largest European island and the ninth-largest island in the world.  It is dominated by a maritime climate with narrow temperature differences between seasons. The island of Ireland, with an area 40 per cent that of Great Britain, is to the west - these islands, along with over 1,000 smaller surrounding islands and named substantial rocks, comprise the British Isles archipelago."""
question = "What is another name for Great Britain?"
answer = question_answer(question, text)  # original answer from the dataset

translator = Translator()
translated_question = translator.translate(question, dest="ro")
translated_answer = translator.translate(answer, dest="ro")

print("Translated question:\n", translated_question.text)
print("Answer:\n", answer)
print("Translated answer:\n", translated_answer.text)
