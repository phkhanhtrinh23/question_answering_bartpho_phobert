import argparse
from utils.metric import *
from datasets import load_dataset
from transformers.models.bartpho.tokenization_bartpho_fast import BartphoTokenizerFast
from transformers import AutoModelForQuestionAnswering, default_data_collator, get_scheduler
from torch import nn
import evaluate
import numpy as np
from torch.optim import AdamW
from tqdm.auto import tqdm
import torch
from torch.utils.data import DataLoader


def preprocess_training_dataset(examples):

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

    offset_mapping = inputs.pop("offset_mapping")
    sample_map = inputs.pop("overflow_to_sample_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        if len(answers[sample_idx]["text"]) > 0:
            answer = answers[sample_idx]
            start_char = answer["answer_start"][0]
            end_char = answer["answer_start"][0] + len(answer["text"][0])
            sequence_ids = inputs.sequence_ids(i)

            # Find the start and end of the context
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            # If the answer is not fully inside the context, label is (0, 0)
            if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
                start_positions.append(0)
                end_positions.append(0)
            else:
                # Otherwise it's the start and end token positions
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)

                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)
        else:
            start_positions.append(0)
            end_positions.append(0)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs


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


def main(raw_datasets, args):

    train_dataset = raw_datasets["train"].map(
        preprocess_training_dataset,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
    )

    validation_dataset = raw_datasets["validation"].map(
        preprocess_validation_dataset,
        batched=True,
        remove_columns=raw_datasets["validation"].column_names,
    )
   
    metric = evaluate.load(args.metric)

    train_dataset.set_format("torch")
    validation_set = validation_dataset.remove_columns(["example_id", "offset_mapping"])
    validation_set.set_format("torch")

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=default_data_collator,
        batch_size=args.batch_size,
    )
    eval_dataloader = DataLoader(
        validation_set,
        collate_fn=default_data_collator,
        batch_size=args.batch_size
    )

    device = torch.device(args.device)
    model = AutoModelForQuestionAnswering.from_pretrained(args.pretrained_model)
    
    # Utilize 2 or more GPUs for training
    if device is torch.device("cuda"):
        model = nn.DataParallel(model) 
    
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr)

    num_update_steps_per_epoch = len(train_dataloader)
    num_training_steps = args.epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        args.scheduler,
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    progress_bar = tqdm(range(num_training_steps))

    prev_metrics = None
    for epoch in range(args.epochs):
        # Training
        model.train()
        for _, batch in enumerate(train_dataloader): # Evaluate after each epoch, not after a number of steps!
            outputs = model(**batch)
            loss = outputs.loss

            # backpropagation in 2 GPUs so we need to calculate mean of loss
            loss.mean().backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

        # Evaluation
        model.eval()
        start_logits = []
        end_logits = []
        print("Evaluation!")
        for batch in tqdm(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)

            start_logits.append(outputs.start_logits.cpu().numpy())
            end_logits.append(outputs.end_logits.cpu().numpy())

        start_logits = np.concatenate(start_logits)
        end_logits = np.concatenate(end_logits)
        start_logits = start_logits[: len(validation_dataset)]
        end_logits = end_logits[: len(validation_dataset)]

        metrics = compute_metrics(
            args, metric, start_logits, end_logits, validation_dataset, raw_datasets["validation"]
        )
        print(f"Epoch {epoch}:", metrics)

        if epoch == 0:
            prev_metrics = metrics
        elif metrics['f1'] > prev_metrics['f1']:
            print(f"Saving model to {args.output_dir}...")
            model.module.save_pretrained(args.output_dir)
            print("Finished.")
            prev_metrics = metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-metric', type=str, default="squad")
    parser.add_argument('-device', type=str, default="cuda")
    parser.add_argument('-output_dir', type=str, default="checkpoints")
    parser.add_argument('-scheduler', type=str, default="linear")
    parser.add_argument('-pretrained_model', type=str, default="vinai/bartpho-syllable")

    parser.add_argument('-batch_size', type=int, default=2)
    parser.add_argument('-epochs', type=int, default=15)
    parser.add_argument('-lr', type=int, default=2e-5)

    parser.add_argument('-max_length', type=int, default=1024)
    parser.add_argument('-stride', type=int, default=128)

    parser.add_argument('-n_best', type=int, default=20)
    parser.add_argument('-max_answer_length', type=int, default=200)

    args = parser.parse_args()
    
    raw_datasets = load_dataset("utils/viquad.py")

    # Filter examples which have just 1 element in list of 'text' answer
    raw_datasets["validation"] = raw_datasets["validation"].filter(lambda x: len(x["answers"]["text"]) == 1)

    tokenizer = BartphoTokenizerFast.from_pretrained(args.pretrained_model)
    max_length = args.max_length
    stride = args.stride

    main(raw_datasets, args)