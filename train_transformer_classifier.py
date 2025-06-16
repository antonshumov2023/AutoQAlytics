#!/usr/bin/env python3
"""
train_transformer_classifier.py

Fine-tune a BERT-like model for issue classification using HuggingFace Trainer.

Usage:
    pip install transformers datasets sklearn
    python train_transformer_classifier.py \
        --input issues_dataset.jsonl \
        --target severity \
        --model_name bert-base-uncased \
        --output_dir ./tf_model_severity
"""

import argparse
from datasets import load_dataset, ClassLabel, DatasetDict
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, DataCollatorWithPadding
)
from sklearn.preprocessing import LabelEncoder
import numpy as np
import evaluate
import pandas as pd
from datasets import Dataset

def main(args):
    # 1. Read JSONL via pandas
    df = pd.read_json(args.input, lines=True)

    # 2. Encode the target into ints
    le = LabelEncoder()
    df["label"] = le.fit_transform(df[args.target].fillna("Unknown"))

    # 3. Build HF Dataset and stratified split
    dataset = Dataset.from_pandas(df, preserve_index=False)
    dataset = dataset.class_encode_column("label")
    ds = dataset.train_test_split(
        test_size=0.2,
        stratify_by_column="label",
        seed=42
    )

    # 4. Tokenizer + model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(le.classes_))

    # 5. Preprocess function
    def preprocess(batch):
        tokens = tokenizer(batch['clean_body'], truncation=True, padding=True, max_length=256)
        tokens['label'] = batch["label"]
        return tokens

    tokenized = ds.map(preprocess, batched=True, remove_columns=ds['train'].column_names)

    # 6. Metrics
    accuracy = evaluate.load('accuracy')
    f1_micro = evaluate.load('f1')

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            'accuracy': accuracy.compute(predictions=preds, references=labels)['accuracy'],
            'f1_micro': f1_micro.compute(predictions=preds, references=labels, average='micro')['f1']
        }

    # 7. Data collator & training args
    data_collator = DataCollatorWithPadding(tokenizer)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy='epoch',
        save_strategy='epoch',
        logging_steps=50,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        learning_rate=2e-5,
        load_best_model_at_end=True,
        metric_for_best_model='f1_micro',
    )

    # 8. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized['train'],
        eval_dataset=tokenized['test'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # 9. Train & save
    trainer.train()
    trainer.save_model(args.output_dir)
    # Save label encoder
    import pickle
    with open(f"{args.output_dir}/label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)
    print(f"Model and encoder saved to {args.output_dir}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input",      required=True, help="JSONL dataset path")
    p.add_argument("--target",     required=True,
                   choices=['severity','category'], help="Label to predict")
    p.add_argument("--model_name", default="bert-base-uncased", help="HuggingFace model")
    p.add_argument("--output_dir", required=True, help="Where to save checkpoints")
    args = p.parse_args()
    main(args)
