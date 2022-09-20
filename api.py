from flask import Flask, render_template, request, url_for, redirect
from datasets import load_dataset
from transformers.models.bartpho.tokenization_bartpho_fast import BartphoTokenizerFast
from transformers import AutoModelForQuestionAnswering
import torch
import json
import os
import collections
import numpy as np
from torch import nn


app = Flask(__name__)

# Example
messages = [
        {
            'question': 'This is the question 1.',
            'context': 'This is the context 1.',
            'answer': 'This is the answer 1.',
        },
            ]

global_model = None


def preprocess_validation_dataset(examples):

    questions = [q.strip() for q in examples["question"]]
    contexts = [c.strip() for c in examples["context"]]
    inputs = tokenizer(
        questions,
        contexts,
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_map = inputs.pop("overflow_to_sample_mapping")
    example_ids = []

    for i in range(len(inputs["input_ids"])):
        sample_idx = sample_map[i]
        example_ids.append(examples["id"][sample_idx])

        sequence_ids = inputs.sequence_ids(i)
        offset = inputs["offset_mapping"][i]
        inputs["offset_mapping"][i] = [
            o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
        ]

    inputs["example_id"] = example_ids
    return inputs

class ViQuADModel:

    def __init__(self, device, checkpoints, n_best, max_answer_length):

        self.device = torch.device(device)
        self.checkpoints = checkpoints

        self.n_best = n_best
        self.max_answer_length = max_answer_length

        self.model = AutoModelForQuestionAnswering.from_pretrained(self.checkpoints)
        self.model = nn.DataParallel(self.model)
        self.model.to(self.device)
        

    def forward(self, raw_datasets):

        eval_set = raw_datasets["test"].map(
            preprocess_validation_dataset,
            batched=True,
            remove_columns=raw_datasets["test"].column_names,
            load_from_cache_file=False
        )

        eval_set_for_model = eval_set.remove_columns(["example_id", "offset_mapping"])
        eval_set_for_model.set_format("torch")

        example_to_features = collections.defaultdict(list)
        for idx, feature in enumerate(eval_set):
            example_to_features[feature["example_id"]].append(idx)

        batch = {k: eval_set_for_model[k].to(self.device) for k in eval_set_for_model.column_names}

        with torch.no_grad():
            outputs = self.model(**batch)

        start_logits = outputs.start_logits.cpu().numpy()
        end_logits = outputs.end_logits.cpu().numpy()

        example = raw_datasets["test"][0]
        example_id = example["id"]
        context = example["context"]
        answers = []

        for feature_index in example_to_features[example_id]:
            
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offsets = eval_set["offset_mapping"][feature_index]

            start_indexes = np.argsort(start_logit)[-1 : -self.n_best - 1 : -1].tolist()
            end_indexes = np.argsort(end_logit)[-1 : -self.n_best - 1 : -1].tolist()

            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Skip answers that are not fully in the context
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue

                    # Skip answers with a length that is either < 0 or > max_answer_length.
                    if (
                        end_index < start_index
                        or end_index - start_index + 1 > self.max_answer_length
                    ):
                        continue

                    answers.append(
                        {
                            "text": context[offsets[start_index][0] : offsets[end_index][1]],
                            "logit_score": start_logit[start_index] + end_logit[end_index],
                        }
                    )

        best_answer = max(answers, key=lambda x: x["logit_score"])

        return best_answer["text"].strip()


@app.route('/')
def index():
    return render_template('index.html', messages=messages)


@app.route('/create/', methods=('GET', 'POST'))
def create():
    if request.method == 'POST':
        question = request.form['question']
        context = request.form['context']

        with open("data/demo.json", "r", encoding="utf-8") as f:
            demo_data = json.load(f)
        f.close()

        demo_data["data"][0]["paragraphs"][0]["qas"][0]["question"] = question
        demo_data["data"][0]["paragraphs"][0]["context"] = context

        # Save the most recent record at "data/demo.json" file
        with open(os.path.join("data/demo.json"), "w") as f:
            json.dump(demo_data, f, indent= 4, ensure_ascii=False)
        f.close()

        # Load dataset WITHOUT USING CACHE
        raw_datasets = load_dataset("utils/viquad_demo.py", download_mode="force_redownload")
        answer = global_model.forward(raw_datasets)

        if not question or not context:
            return render_template('create.html')
        else:
            messages.append({'question': question, 'context': context, 'answer': answer})
            return redirect(url_for('index'))

    return render_template('create.html')


if __name__ == "__main__":
    global_model = ViQuADModel(
        device="cuda",
        checkpoints="checkpoints",
        n_best=20,
        max_answer_length=200,
    )

    tokenizer = BartphoTokenizerFast.from_pretrained("vinai/bartpho-syllable")
    max_length = 1024
    stride = 128
    
    app.run(debug=True,host="0.0.0.0")