export CUDA_VISIBLE_DEVICES=3,2

task=MMLU
rm=test
models=InternLM7b+qwen4b
em=tas2+mas2

res_path=./res/${task}/${rm}/${models}/${em}
log_path=./log/${task}/${rm}/${models}/${em}
mkdir -vp ${res_path}
mkdir -vp ${log_path}

nohup python unite_mmlu.py --config confs/${task}/${models}.json -rm ${rm} -em ${em} -rsd ${res_path} > ${log_path}/run.log 2>&1 &
