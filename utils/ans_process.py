import json
import re

# GSM
def gsm_parse_pred_ans(filename):
    total, correct = 0, 0
    gold_ans = []
    with open(filename, "r", encoding="utf-8") as fr:
        for line in fr:
            jo = json.loads(line.strip())
            if jo["original_sln"] not in gold_ans:
                correct += jo["pred"] == jo["label"]
                total += 1
                gold_ans.append(jo["original_sln"])
            else:
                continue
    print('num_q %d correct %d ratio %.4f' % (total, correct, float(correct / total)))

# ARC/PIQA/MMLU
def arc_parse_pred_ans(filename):
    total, correct = 0, 0
    gold_ans = []
    qs = []
    with open(filename, "r", encoding="utf-8") as fr:
        for line in fr:
            jo = json.loads(line.strip())
            if jo["question"] not in qs:
                correct += jo["pred"].strip() == jo["label"].strip()
                total += 1
                qs.append(jo["question"])
            else:
                continue
    print('num_q %d correct %d ratio %.4f' % (total, correct, float(correct / total)))
    return float(correct / total)

#TriviaQA NQ
def qa_parse_pred_ans(filename):
    total, correct = 0, 0
    with open(filename, "r", encoding="utf-8") as fr:
        for line in fr:
            jo = json.loads(line.strip())
            for gold in jo["label"]:
                if jo['pred'][:-1].strip() in gold:
                    correct += 1
                    break
            total += 1
    print('num_q %d correct %d ratio %.4f' % (total, correct, float(correct / total)))

