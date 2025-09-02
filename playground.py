from utils.ans_process import *


if __name__ == "__main__":
    # gsm_parse_pred_ans("predictions_gsm8k.jsonl")
    # arc_parse_pred_ans("predictions_arc.jsonl")

    output_file = "res/NQ/dev/OpenChat+LLaMA/vanilla/pred.jsonl"
    qa_parse_pred_ans(output_file)
