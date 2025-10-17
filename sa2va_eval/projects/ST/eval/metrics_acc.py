import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='eval cider and meteor')
    parser.add_argument('--pred', help='pred file')
    args = parser.parse_args()
    return args

args = parse_args()
with open(args.pred, "r") as f:
    pred_items = json.load(f)

all = 0
correct = 0
for pred_item in pred_items:
    pred = pred_item["pred_answer"]
    gt = pred_item["gt_answer"]
    if gt in pred:
        correct += 1
    all += 1
print(f"{correct}/{all}")
