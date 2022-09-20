import argparse
import torch
from transformers import AutoModelForQuestionAnswering
from transformers.models.bartpho.tokenization_bartpho_fast import BartphoTokenizerFast
import collections
import numpy as np
from torch import nn
import evaluate

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


def inference(raw_datasets, args):

    num_batch = len(raw_datasets["validation"]) // args.batch + 1
    
    for i in range(num_batch):
        
        if i*args.batch > len(raw_datasets["validation"]) - args.batch:
            print("Loading: ", (i*args.batch, len(raw_datasets["validation"])), "...")
            batch_eval_set = raw_datasets["validation"].select(range(i*args.batch, len(raw_datasets["validation"])))
            
        else:
            print("Loading: ", (i*args.batch, i*args.batch + args.batch), "...")
            batch_eval_set = raw_datasets["validation"].select(range(i*args.batch, i*args.batch + args.batch))

        eval_set = batch_eval_set.map(
            preprocess_validation_dataset,
            batched=True,
            remove_columns=raw_datasets["validation"].column_names,
        )

        eval_set_for_model = eval_set.remove_columns(["example_id", "offset_mapping"])
        eval_set_for_model.set_format("torch")
        
        device = torch.device(args.device)
        batch = {k: eval_set_for_model[k].to(device) for k in eval_set_for_model.column_names}
        model = AutoModelForQuestionAnswering.from_pretrained(args.checkpoints)
        
        # Utilize 2 or more GPUs for training
        if device is torch.device("cuda"):
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
            predicted_answers.append({"id": example_id, "prediction_text": best_answer["text"]})

        metric = evaluate.load(args.metric)

        theoretical_answers = [
            {"id": ex["id"], "answers": ex["answers"]} for ex in batch_eval_set
        ]

        for i in range(len(predicted_answers)):
            print("Context: ", contexts[i])
            print("Question: ", questions[i])
            print("Answer: ", predicted_answers[i])
            print("Label: ", theoretical_answers[i])

            print(metric.compute(predictions=[predicted_answers[i]], references=[theoretical_answers[i]]))
            print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-metric', type=str, default="squad")
    parser.add_argument('-device', type=str, default="cuda")
    parser.add_argument('-scheduler', type=str, default="linear")
    parser.add_argument('-pretrained_model', type=str, default="vinai/bartpho-syllable")

    parser.add_argument('-checkpoints', type=str, default="checkpoints")

    parser.add_argument('-max_length', type=int, default=1024)
    parser.add_argument('-stride', type=int, default=128)

    parser.add_argument('-n_best', type=int, default=20)
    parser.add_argument('-max_answer_length', type=int, default=30)

    parser.add_argument('-samples', type=int, default=20)

    # We cannot infer all samples in dataset so we use batch inference
    parser.add_argument('-batch', type=int, default=20)

    args = parser.parse_args()
    
    raw_datasets = load_dataset("utils/viquad.py")

    # Filter examples which have just 1 element in list of 'text' answer
    raw_datasets["validation"] = raw_datasets["validation"].filter(lambda x: len(x["answers"]["text"]) == 1)

    tokenizer = BartphoTokenizerFast.from_pretrained(args.pretrained_model)
    max_length = args.max_length
    stride = args.stride

    inference(raw_datasets, args)