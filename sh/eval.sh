task=MMLU
rm=test
models=InternLM7b+qwen4b

python utils/evaluate/EM_dir_test.py res/${task}/${rm}/${models}/vanilla
# python utils/evaluate/EM_dir_test.py res/${task}/${rm}/${models}/tas
# python utils/evaluate/EM_dir_test.py res/${task}/${rm}/${models}/tas2
python utils/evaluate/EM_dir_test.py res/${task}/${rm}/${models}/tas2+mas2
python utils/evaluate/EM_dir_test.py res/${task}/${rm}/${models}/tas3+mas2
