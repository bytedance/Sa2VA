import json

import nltk
from nltk.translate.meteor_score import meteor_score
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from nltk.translate.bleu_score import sentence_bleu
import argparse
import os

nltk.download('wordnet')
nltk.download('omw-1.4')

def parse_args():
    parser = argparse.ArgumentParser(description='eval cider and meteor')
    parser.add_argument('--pred', help='pred file')
    args = parser.parse_args()
    return args

args = parse_args()
with open(args.pred, "r") as f:
    pred_items = json.load(f)

METEOR_scores = []
BLUE_scores = []
for pred_item in pred_items:
    reference = pred_item["pred_caption"]
    hypothesis = pred_item["gt_caption"]
    score = meteor_score([reference.split()], hypothesis.split())
    bleu_score = sentence_bleu([reference.split()], hypothesis.split())
    BLUE_scores.append(bleu_score)
    METEOR_scores.append(score)
print(f"BLEU score: {sum(BLUE_scores)/ len(BLUE_scores)}")
print(f"METEOR score: {sum(METEOR_scores) / len(METEOR_scores)}")


references = {}
hypothesises = {}
for i, pred_item in enumerate(pred_items):
    reference = pred_item["pred_caption"]
    hypothesis = pred_item["gt_caption"]
    references[str(i)] = [{'caption': reference}]
    hypothesises[str(i)] = [{'caption': hypothesis}]

tokenizer = PTBTokenizer()
refs = tokenizer.tokenize(references)
hyps = tokenizer.tokenize(hypothesises)

cider_scorer = Cider()
score, _ = cider_scorer.compute_score(refs, hyps)

print(f"CIDEr score: {score}")

