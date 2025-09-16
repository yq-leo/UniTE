from utils.ans_process import *
import json


if __name__ == "__main__":
    # gsm_parse_pred_ans("predictions_gsm8k.jsonl")
    # arc_parse_pred_ans("predictions_arc.jsonl")

    # output_file = "archive_res/PIQA/dev/OpenChat+LLaMA/vanilla/pred.jsonl"

    # qa_parse_pred_ans(output_file)
    # arc_parse_pred_ans(output_file)

    
    input_file = "/home/qiyu6/UniTE/res/NQ/test/OpenChat+InternLM7b+LLaMA3/tas2+mas2/pred.jsonl"
    output_file = "/home/qiyu6/UniTE/res/NQ/test/OpenChat+InternLM7b+LLaMA3/tas2+mas2/pred_processed.jsonl"

    with open(input_file, "r", encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            obj = json.loads(line)

            # Duplicate keys
            obj["answer"] = obj.get("original_sln")
            obj["prediction"] = obj.get("pred")

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
