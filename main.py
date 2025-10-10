import random
from src.medrgag import MedRGAG
import os
import json
import argparse
from src.evaluate_utils import QADataset, locate_answer, get_acc
from tqdm import tqdm
from time import time
import sys
from src.utils import read_json
from src.llm_by_gpt import Function_LLM
import numpy as np

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--reader_name", type=str, default="qwen2.5") ## ministral-8B llama3.1 qwen2.5
    parser.add_argument("--data_name", type=str, default="medqa")  # medqa medmcqa mmlu pubmedqa bioasq
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--retriever_name", type=str, default="BM25") # BM25
    parser.add_argument("--corpus_name", type=str, default="MedText")  # MedText
    parser.add_argument("--results_dir", type=str, default="./outputs")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
    reader_name = args.reader_name
    print(f">>>>>Current Reader is {reader_name}!")
    topk = args.k
    corpus_name = args.corpus_name
    retriever_name = args.retriever_name
    results_dir = args.results_dir
    medrgag = MedRGAG(reader_name=reader_name, retriever_name=retriever_name, corpus_name=corpus_name, HNSW=False)
    dataset_names = [args.data_name]
    datasets = {key: QADataset(key) for key in dataset_names}
    avg_scores = []
    t0 = time()
    generator_name = "llama3.1" #  llama3.1 qwen2.5-14b
    summary_llm = explore_llm = selection_llm= "gpt-4o-mini-2024-07-18"
    print("Start time: {:.2f} seconds".format(t0))
    medrgag.init_ranker("medcpt-rank")
    for dataset_name in dataset_names:
        print("[{:s}] ".format(dataset_name), end="")
        locate_fun = locate_answer
        split = "test"
        if dataset_name == "medmcqa":
            split = "dev"
        save_dir = os.path.join(results_dir, dataset_name, "prediction", llm_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        retrieval_dir = os.path.join(results_dir, dataset_name, "docs", "retrieval")
        generator_dir = os.path.join(results_dir, dataset_name, "docs", "generate")
        if not os.path.exists(retrieval_dir):
            os.makedirs(retrieval_dir)
        if not os.path.exists(generator_dir):
            os.makedirs(generator_dir)
        ret_path = os.path.join(retrieval_dir, f"snippets.json")
        ret_k = 32
        if os.path.exists(ret_path):
            retrieved_snippets = read_json(ret_path)
            retrieved_flag = False
        else:
            retrieved_flag = True
            retrieved_snippets = {}
            for id, data in tqdm(datasets[dataset_name][:]):  ## 10178//2
                question = data["question"]
                options = data["options"]
                snippets, _ = medrgag.retrieval(question=question, options=options, k=ret_k)
                snippets, scores = medrgag.rank_snippets(question=question, options=options, snippets=snippets, mode="score")
                retrieved_snippets[id] = {
                    "id": id,
                    "snippets": snippets,
                    "scores": scores
                }
            with open(ret_path, 'w') as f:
                json.dump(retrieved_snippets, f, indent=4)
            print(f"Retrieval cost {(time() - t0) / 60:.2f} minutes")
        gen_path = os.path.join(generator_dir, f"snippets.json")
        if os.path.exists(gen_path):
            generated_snippets = read_json(gen_path)
            generated_flag = False
        else:
            generated_flag = True
            summary_snippets = Function_LLM(mode="summary", data_name=dataset_name, model_name=summary_llm)
            explore_snippets = Function_LLM(mode="explore", data_name=dataset_name, model_name=explore_llm)
            batch_data = []
            for id, data in tqdm(datasets[dataset_name][:]): ## 10178//2
                question = data["question"]
                options = data["options"]
                batch_data.append({
                    "id": id,
                    "question": question,
                    "options": options,
                    "explore_contents": explore_snippets[data["id"]],
                })
            generated_snippet = medrgag.generate_knowledge(batch_data, generator_name)
            generated_snippets = {}
            for id, data in tqdm(datasets[dataset_name][:]):
                question = data["question"]
                options = data["options"]
                snippets = [{
                    "id": generator_name + f"_{j}",
                    "contents": v
                } for j, v in enumerate(generated_snippet[id][:5])]
                item = {
                    "id": id,
                    "snippets": snippets,
                    "scores": [],
                }
                generated_snippets[str(id)] = item
            with open(gen_path, 'w') as f:
                json.dump(generated_snippets, f, indent=4)
            print(f"Generator cost {(time() - t0) / 60:.2f} minutes")
        select_path = os.path.join(generator_dir, f"snippets_final_select.json")
        if os.path.exists(select_path):
            select_snippets = read_json(select_path)
            selected_flag = False
        else:
            select_snippets = Function_LLM(mode="selection", data_name=dataset_name, model_name=selection_llm)

        batch_data = []
        ret_ratios = []
        for id, data in tqdm(datasets[dataset_name][:]): ## 10178//2
            question = data["question"]
            options = data["options"]
            gt_answer = data["answer"]
            ret_snippets = retrieved_snippets[id]["snippets"][:topk]
            gen_snippets = generated_snippets[id]["snippets"][:topk]
            snippets = ret_snippets + gen_snippets
            select_list = select_snippets[id]["final_list"]
            snippets = [snippets[num] for num in select_list]
            ret_ratio = [1 if pid < 5 else 0 for pid in select_list]
            snippets, _ = medrgag.rank_snippets(question=question, options=options, snippets=snippets,
                                                       mode="select")
            snippets = snippets[:topk]
            ret_ratios.append(np.mean(ret_ratio[:topk]))
            batch_data.append({
                "id": id,
                "question": question,
                "options": options,
                "gt_answer": gt_answer,
                "snippets": snippets,
            })
        results = []
        pre_answer_dict = medrgag.vllm_answer(batch_data)
        for data in batch_data:
            pre_answer = pre_answer_dict[data["id"]]
            model_answer = locate_fun(pre_answer.split('"answer_choice": "')[-1].strip())
            item = {"id": data["id"], "question": data["question"], "options": data["options"], "answer": data["gt_answer"],
                    "model_answer": model_answer, "model_prediction": pre_answer}
            results.append(item)
        with open(os.path.join(save_dir, f"prediction.json"), 'w') as f:
            json.dump(results, f, indent=4)
        acc, std, flag = get_acc(results)
        avg_scores.append(acc)
        print("mean acc: {:.4f}; proportion std: {:.4f}".format(acc, std), end="")
        if flag:
            print(" (NOT COMPLETED)")
        else:
            print("")

    if len(avg_scores) > 0:
        print("[Average] mean acc: {:.4f}".format(sum(avg_scores) / len(avg_scores)))

    t1 = time()
    print(f"Total cost {(t1-t0)/60:.2f} minutes")