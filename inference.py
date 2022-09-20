import argparse
import json
import os
import torch
from transformers import AutoModelForQuestionAnswering
from transformers.models.bartpho.tokenization_bartpho_fast import BartphoTokenizerFast
import collections
import numpy as np
from torch import nn

from datasets import load_dataset


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


def inference(custom_dataset, args):
    
    num_batch = len(custom_dataset["test"]) // args.batch + 1

    results = {}
    
    for i in range(num_batch):
        
        if i*args.batch > len(custom_dataset["test"]) - args.batch:
            print("Loading: ", (i*args.batch, len(custom_dataset["test"])), "...")
            batch_eval_set = custom_dataset["test"].select(range(i*args.batch, len(custom_dataset["test"])))
            
        else:
            print("Loading: ", (i*args.batch, i*args.batch + args.batch), "...")
            batch_eval_set = custom_dataset["test"].select(range(i*args.batch, i*args.batch + args.batch))

        eval_set = batch_eval_set.map(
            preprocess_validation_dataset,
            batched=True,
            remove_columns=batch_eval_set.column_names,
        )

        eval_set_for_model = eval_set.remove_columns(["example_id", "offset_mapping"])
        eval_set_for_model.set_format("torch")

        device = torch.device(args.device)
        batch = {k: eval_set_for_model[k].to(device) for k in eval_set_for_model.column_names}
        model = AutoModelForQuestionAnswering.from_pretrained(args.checkpoints)
        
        # Utilize 2 or more GPUs for training
        if device == torch.device("cuda"):
            model = nn.DataParallel(model)
        
        model.to(device)

        example_to_features = collections.defaultdict(list)
        for idx, feature in enumerate(eval_set):
            example_to_features[feature["example_id"]].append(idx)

        with torch.no_grad():
            outputs = model(**batch)

        start_logits = outputs.start_logits.cpu().numpy()
        end_logits = outputs.end_logits.cpu().numpy()

        predicted_answers = []
        questions = []
        contexts = []

        for example in batch_eval_set:
            example_id = example["id"]
            context = example["context"]
            answers = []
            questions.append(example["question"])
            contexts.append(example["context"])

            for feature_index in example_to_features[example_id]:
                start_logit = start_logits[feature_index]
                end_logit = end_logits[feature_index]
                offsets = eval_set["offset_mapping"][feature_index]

                start_indexes = np.argsort(start_logit)[-1 : -args.n_best - 1 : -1].tolist()
                end_indexes = np.argsort(end_logit)[-1 : -args.n_best - 1 : -1].tolist()

                for start_index in start_indexes:
                    for end_index in end_indexes:

                        if offsets[start_index] is None or offsets[end_index] is None:
                            continue

                        if (
                            end_index < start_index
                            or end_index - start_index + 1 > args.max_answer_length
                        ):
                            continue

                        answers.append(
                            {
                                "text": context[offsets[start_index][0] : offsets[end_index][1]],
                                "logit_score": start_logit[start_index] + end_logit[end_index],
                            }
                        )

            best_answer = max(answers, key=lambda x: x["logit_score"])
            predicted_answers.append({"id": example_id, "prediction_text": best_answer["text"].strip()})

        
        for answer in predicted_answers:
            # if answer["id"] in results:
            #     # print("Duplicate Id!")
            #     if isinstance(results[answer["id"]], str):
            #         results[answer["id"]] = [results[answer["id"]], answer["prediction_text"]]
            #     else:
            #         results[answer["id"]].append(answer["prediction_text"])
            # else:
            #     results[answer["id"]] = answer["prediction_text"]
            results[answer["id"]] = answer["prediction_text"]


    with open(os.path.join("results.json"), "w") as f:
        json.dump(results, f, indent= 4, ensure_ascii=False)
    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-device', type=str, default="cuda")
    parser.add_argument('-pretrained_model', type=str, default="vinai/bartpho-syllable")

    parser.add_argument('-checkpoints', type=str, default="checkpoints")

    parser.add_argument('-max_length', type=int, default=1024)
    parser.add_argument('-stride', type=int, default=128)

    parser.add_argument('-n_best', type=int, default=20)
    parser.add_argument('-max_answer_length', type=int, default=200)
    
    # We cannot infer all samples in dataset so we use batch inference
    parser.add_argument('-batch', type=int, default=20)

    args = parser.parse_args()

    tokenizer = BartphoTokenizerFast.from_pretrained(args.pretrained_model)
    max_length = args.max_length
    stride = args.stride

    custom_dataset = load_dataset("utils/viquad_test.py", download_mode="force_redownload")

    inference(custom_dataset, args)