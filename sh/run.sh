export CUDA_VISIBLE_DEVICES=2,3

task=GSM8K
rm=test
models=qwen4b+InternLM7b
em=tas3+mas2

res_path=./res/${task}/${rm}/${models}/${em}
log_path=./log/${task}/${rm}/${models}/${em}
mkdir -vp ${res_path}
mkdir -vp ${log_path}

nohup python unite2.py --config confs/${task}/${models}.json -rm ${rm} -em ${em} -rsd ${res_path} > ${log_path}/run.log 2>&1 &
