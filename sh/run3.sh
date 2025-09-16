export CUDA_VISIBLE_DEVICES=0,1,2

task=TriviaQA
rm=test
models=OpenChat+InternLM7b+LLaMA3
em=tas3+mas2

res_path=./res/${task}/${rm}/${models}/${em}
log_path=./log/${task}/${rm}/${models}/${em}
mkdir -vp ${res_path}
mkdir -vp ${log_path}

nohup python unite3.py --config confs/${task}/${models}.json -rm ${rm} -em ${em} -rsd ${res_path} > ${log_path}/run.log 2>&1 &
