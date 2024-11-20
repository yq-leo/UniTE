# UniTE (Determine-Then-Ensemble: Necessity of Top-k Union for Large Language Model Ensembling)

Official code of the paper [Determine-Then-Ensemble: Necessity of Top-k Union for Large Language Model Ensembling](https://arxiv.org/abs/2410.03777)

## Introduction

Large language models (LLMs) exhibit varying strengths and weaknesses across different tasks, prompting recent studies to explore the benefits of model ensembling to leverage their complementary advantages. However, existing LLM ensembling methods often overlook model compatibility and struggle with inefficient probability alignment across the entire vocabulary.

In this work, we empirically investigate the factors influencing ensemble performance, identifying model performance, vocabulary size, and response style as key determinants. Our analysis reveals that compatibility among models is essential for effective ensembling, leading to the development of a straightforward yet effective model selection strategy to identify compatible models.

Figure 1 shows the impact of performance disparity among models on ensemble effectiveness across different datasets and methods.

Figure 2 represents the impact of performance differences on model ensembling effectiveness on GSM8K dataset. OOM represents out of memory issue.

We introduce the UNIon Top-k Ensembling (UNITE), a novel approach that efficiently combines models by focusing on the union of the top-k tokens from each model. This method avoids the need for full vocabulary alignment, significantly reducing computational overhead. Extensive evaluations across multiple benchmarks demonstrate that UNITE significantly enhances performance compared to existing methods, providing a more efficient framework for LLM ensembling.

## Usage
For 2-model ensembling: run ```unite2.py```

For 3- model ensembling: run ```unite3.py```

## Citation
```
@misc{yao2024determinethenensemblenecessitytopkunion,
      title={Determine-Then-Ensemble: Necessity of Top-k Union for Large Language Model Ensembling}, 
      author={Yuxuan Yao and Han Wu and Mingyang Liu and Sichun Luo and Xiongwei Han and Jie Liu and Zhijiang Guo and Linqi Song},
      year={2024},
      eprint={2410.03777},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2410.03777}, 
}
```
