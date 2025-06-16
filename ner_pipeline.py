#!/usr/bin/env python3
"""
ner_pipeline.py

1) Build a hybrid NER pipeline with spaCy:
   – EntityRuler for rule-based patterns (ERROR_CODE, OS, BROWSER…)
   – Transformer-powered NER for more flexible, learned entities (MODULE, CUSTOMER_ID…)

2) Optionally fine-tune the NER model on the annotated examples.

Usage:
    # 1. Install dependencies
    pip install spacy spacy-transformers

    # 2. Download base model
    python -m spacy download en_core_web_trf

    # 3a. To train:
    python ner_pipeline.py --train --input train_data.spacy --output models/qa_ner

    # 3b. To extract from a raw text file:
    python ner_pipeline.py --extract --model models/qa_ner --text "ERROR_777 in Module KERNEL on Windows 12"
"""

import random
import spacy
import argparse
from spacy import displacy
from spacy.tokens import DocBin
from spacy.training import Example
from spacy.util import minibatch, compounding


def build_pipeline(model_name="en_core_web_trf"):
    """Load base model, add EntityRuler + NER (if not present)."""
    nlp = spacy.load(model_name)

    # 1) RULE-BASED: add EntityRuler before the NER component
    patterns = [
        {"label": "ERROR_CODE", "pattern": [{"TEXT": {"REGEX": r"ERR_\d+"}}]},
        {"label": "OS", "pattern": [{"LOWER": {"IN": ["windows", "linux", "mac", "ubuntu", "debian"]}}]},
        {"label": "BROWSER", "pattern": [{"LOWER": {"IN": ["chrome", "firefox", "safari", "edge"]}}]},
        # e.g. module names: load from your project’s lexicon
        {"label": "MODULE", "pattern": "DeepSpeed"},
        {"label": "MODULE", "pattern": "scheduler"},
        # add more module names or placeholders as needed
    ]
    ruler = nlp.add_pipe("entity_ruler", before="ner")
    ruler.add_patterns(patterns)

    return nlp


def train_ner(input_spacy_path, output_dir, n_iter=20):
    # Load base nlp + patterns
    nlp = build_pipeline()

    # Load your gold docs with .ents set
    db = DocBin().from_disk(input_spacy_path)
    gold_docs = list(db.get_docs(nlp.vocab))

    # Convert to spaCy Example objects
    examples = []
    for gold in gold_docs:
        # create a fresh, un-annotated Doc for prediction
        pred = nlp.make_doc(gold.text)
        examples.append(Example(pred, gold))

    # Ensure NER in pipeline
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner", last=True)
    else:
        ner = nlp.get_pipe("ner")
    # Register any new labels
    for gold in gold_docs:
        for ent in gold.ents:
            ner.add_label(ent.label_)

    optimizer = nlp.resume_training()
    for i in range(n_iter):
        losses = {}
        random.shuffle(examples)  # shuffle Example objects
        batches = minibatch(examples, size=compounding(4.0, 32.0, 1.5))
        for batch in batches:
            # batch is a list of Example objects
            nlp.update(batch, sgd=optimizer, drop=0.2, losses=losses)
        print(f"Epoch {i+1}/{n_iter}  Losses: {losses}")

    nlp.to_disk(output_dir)
    print(f"NER model saved to {output_dir}")


def extract_entities(model_dir, text):
    """Load model and print out extracted entities."""
    nlp = spacy.load(model_dir)
    doc = nlp(text)
    print(f"\nInput: {text}\nExtracted Entities:")
    for ent in doc.ents:
        print(f"  • {ent.text:20} → {ent.label_}")
    # optional HTML visualization
    # displacy.serve(doc, style="ent")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid NER pipeline for QA automation")
    parser.add_argument("--train", action="store_true", help="Train NER on a .spacy file")
    parser.add_argument("--extract", action="store_true", help="Run extraction on a text snippet")
    parser.add_argument("--input", help="Path to input .spacy (for train) or raw text file (for extract)")
    parser.add_argument("--model", default="en_core_web_trf", help="Model dir for extraction or base model for training")
    parser.add_argument("--output", default="models/qa_ner", help="Dir to save trained model")
    parser.add_argument("--text", help="Raw text to extract entities from")
    parser.add_argument("--iters", type=int, default=20, help="Training epochs")
    args = parser.parse_args()

    if args.train:
        if not args.input:
            parser.error("--train requires --input train_data.spacy")
        train_ner(args.input, args.output, n_iter=args.iters)

    elif args.extract:
        if not args.text and not args.input:
            parser.error("--extract requires --text or --input <textfile>")
        snippet = args.text or open(args.input).read()
        extract_entities(args.model, snippet)

    else:
        parser.print_help()
