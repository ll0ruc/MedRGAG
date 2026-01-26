<h1 align="center">From Retrieval to Generation: Unifying External and Parametric Knowledge for Medical Question Answering</h1>


## 🔭 Overview

### From Retrieval to Generation: Unifying External and Parametric Knowledge for Medical Question Answering (Accepted in WWW2026)

 In this work, we propose MedRGAG, a unified retrieval--generation augmented framework that seamlessly integrates external and parametric knowledge for medical QA. 
 
 MedRGAG comprises two key modules: Knowledge-Guided Context Completion (KGCC), which directs the generator to produce background documents that complement the missing knowledge revealed by retrieval; and Knowledge-Aware Document Selection (KADS), which adaptively selects an optimal combination of retrieved and generated documents to form concise yet comprehensive evidence for answer generation. 

 ## ⚙️  Installation
Note that the code in this repo runs under **Linux** system. We have not tested whether it works under other OS.

1. **Clone this repository:**

    ```bash
    cd MedRGAG
    ```

2. **Create and activate the conda environment:**

    ```bash
    conda create -n medrgag python=3.10
    conda activate automir
    pip install torch==2.6.0
    pip install faiss-gpu==1.7.2
    pip install deepspeed==0.17.4
    pip install transformers==44.53.2
    pip install sentence-transformers==5.0.0
    pip install datasets==3.6.0
    pip install vllm==0.8.5
    pip install openai==1.86.0
    ```

## 💽 Evaluate
Run the following command to get results:
  ```bash
  python main.py --reader_name qwen2.5 --data_name medqa
  * `--reader_name`: the reader LLM.
  * `--data_name`: the dataset name.
  ```
You will get the evaluation results in the outputs/ folder, which contains the evaluation results.


## 📜Reference

If this code or dataset contributes to your research, please kindly consider citing our paper and give this repo ⭐️ :)
```
@misc{li2025retrievalgenerationunifyingexternal,
      title={From Retrieval to Generation: Unifying External and Parametric Knowledge for Medical Question Answering}, 
      author={Lei Li and Xiao Zhou and Yingying Zhang and Xian Wu},
      year={2025},
      eprint={2510.18297},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2510.18297}, 
}
```
