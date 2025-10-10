import os
import re
import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, AutoModel
import openai
import tiktoken
import sys
sys.path.append("src")
from utils import RetrievalSystem
from template import *
from vllm import LLM, SamplingParams
from tqdm import tqdm
from collections import defaultdict
from typing import cast, List
import numpy as np
import gc
import ast
import math
from sklearn.metrics.pairwise import cosine_similarity

from config import config
global_num = 0

llm_dict = {
            "qwen2.5": "****/Qwen/Qwen2.5-7B-Instruct",
            "qwen2.5-14b": "****/Qwen/Qwen2.5-14B-Instruct",
            "medcpt-rank": "****/ncbi/MedCPT-Cross-Encoder",
            "bge-rank": "****/BAAI/bge-reranker-large",
            "llama3.1": "****/LLM-Research/Meta-Llama-3.1-8B-Instruct",
            "ministral-8B": "****/mistralai/Ministral-8B-Instruct-2410"
            }


class MedRGAG:
    def __init__(self, reader_name="qwen2.5", retriever_name="BM25", corpus_name="Textbooks", db_dir="./corpus", cache_dir=None, corpus_cache=False, HNSW=False):
        self.reader_name = llm_dict[reader_name]
        self.retriever_name = retriever_name
        self.corpus_name = corpus_name
        self.db_dir = db_dir
        self.cache_dir = cache_dir
        self.docExt = None
        self.retrieval_system = RetrievalSystem(self.retriever_name, self.corpus_name, self.db_dir, cache=corpus_cache, HNSW=HNSW)
        self.templates = {"rag": rag_template, "gen": gen_template,
                          "llm_select": llm_selection_prompt, "gen_wo_know": gen_wo_know_template,
                          "summary": knowledge_summary_template, "explore":knowledge_explore_template,
                        }

        self.max_length = 131072
        self.context_length = 128000
        self.reader_tokenizer = AutoTokenizer.from_pretrained(self.reader_name, cache_dir=self.cache_dir)

    def get_generator_prompt(self, data):
        options = "".join([k + ": " + v + "\n" for k, v in data["options"].items()]).strip()
        contexts = ["Knowledge: {:s}".format(data["explore_contents"][i].replace("\n", " ")) for i in
                    range(len(data["explore_contents"]))]
        prompts = [self.templates["gen"].format(question=data["question"] + "\n" + options, content=context) for
                   context
                   in contexts]
        if len(prompts) < 5:
            prompts += [self.templates["gen_wo_know"].format(question=data["question"] + "\n" + options)] * (5 - len(prompts))
        assert len(prompts) == 5
        return prompts

    def clean_generated_text(self, generated_text):
        generated_text = generated_text.replace("### Context:", "").replace("###Context:", "")
        generated_text = generated_text.replace("### Background Document:", "").replace("###Background Document:", "").replace("### Background Document", "").replace("###Background Document", "")
        generated_text = generated_text.replace("Background Document", "").replace("###Background Document:", "")
        output = generated_text.strip()
        if output == "":
            print(f"Clean Error in {generated_text}")
        return output

    def generate_knowledge(self, batch_data, generator_name):
        generator_name = llm_dict[generator_name]
        N = 1
        generator_params = SamplingParams(temperature=1.2, top_p=0.9, top_k=50, max_tokens=256, n=N,
                                          presence_penalty=1.0)
        generator_tokenizer = AutoTokenizer.from_pretrained(generator_name, cache_dir=self.cache_dir)
        generator_model = LLM(model=generator_name, tensor_parallel_size=2, disable_custom_all_reduce=True,
                         trust_remote_code=True, gpu_memory_utilization=0.9)
        print(f">>>>>>generator have downloaded from {generator_name}!")
        results = defaultdict(list)
        batch_size = 64
        batchs = [batch_data[i:i + batch_size] for i in range(0, len(batch_data), batch_size)]
        print(f"一共{len(batch_data)}个query！")
        global_num = 0
        global_num += 1
        print_flag1, print_flag2 = 0, 0
        for batch in tqdm(batchs):
            batch_text = []
            for entry in batch:
                input_prompts = self.get_generator_prompt(entry)
                if print_flag1 == 0 and global_num == 1:
                    print()
                    print(">>>>>>test prompt knowledge 1")
                    print(input_prompts[0])
                print_flag1 += 1
                sys_prompt = "You are a helpful assistant."
                for input_prompt in input_prompts:
                    messages = [
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": input_prompt}
                    ]
                    text = generator_tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    batch_text.append(text)
            outputs = generator_model.generate(batch_text, generator_params, use_tqdm=False)
            batch = [i for i in batch for _ in range(5)]
            for entry, output in zip(batch, outputs):
                generated_texts = [self.clean_generated_text(output.outputs[0].text).strip()]
                if print_flag2 == 0 and global_num == 1:
                    print()
                    print(">>>>>>generated_text 0")
                    print(generated_texts[0])
                    print()
                print_flag2 += 1
                results[entry["id"]].extend(generated_texts)
        del generator_model, generator_tokenizer
        torch.cuda.empty_cache()
        gc.collect()
        return results

    def retrieval(self, question, options=None, k=32, **kwargs):
        if options is not None:
            options = '\n'.join([key + ". " + options[key] for key in sorted(options.keys())])
            question = question + "\n" + options
        # retrieve relevant snippets
        assert self.retrieval_system is not None
        retrieved_snippets, scores = self.retrieval_system.retrieve(question, k=k)
        return retrieved_snippets, scores

    def init_ranker(self, ranker_name):
        self.ranker_name = ranker_name
        rank_model_path = llm_dict[self.ranker_name] #"medcpt-rank", "bge-rank"
        if "bge" in self.ranker_name:
            from FlagEmbedding.inference.reranker.encoder_only.base import BaseReranker
            self.rank_model = BaseReranker(rank_model_path, use_fp16=True, devices=['cuda:0'])
            print(f">>>>>>ranker have downloaded from {rank_model_path}!")
        else:
            self.rank_tokenizer = AutoTokenizer.from_pretrained(rank_model_path)
            self.rank_model = AutoModelForSequenceClassification.from_pretrained(rank_model_path)
            self.rank_batch_size = 8
            self.rank_max_length = 512
            if torch.cuda.is_available():
                self.rank_device = torch.device("cuda")
            else:
                self.rank_device = torch.device("cpu")
            self.rank_model = self.rank_model.to(self.rank_device)
            num_gpus = torch.cuda.device_count()
            if num_gpus > 1:
                self.rank_model = torch.nn.DataParallel(self.rank_model)
                print(f">>>>>>ranker have downloaded from {rank_model_path}!")

    @torch.no_grad()
    def rank_func(self, sentence_pairs: List[List], **kwargs) -> np.ndarray:
        if "bge" in self.ranker_name:
            scores = self.rank_model.compute_score(sentence_pairs)
        else:
            self.rank_model.eval()
            inputs = self.rank_tokenizer(
                sentence_pairs,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=self.rank_max_length,
            ).to(self.rank_device)
            scores = self.rank_model(**inputs).logits.squeeze(dim=1)  # shape (batch_size,)
            scores = scores.cpu().tolist()
        return scores

    def rank_snippets(self, question=None, options=None, snippets=None, mode="score"):
        if options is not None:
            options = '\n'.join([key + ". " + options[key] for key in sorted(options.keys())])
            question = question + "\n" + options
        sentence_pairs = [[question, snippet["contents"]] for snippet in snippets]
        scores = self.rank_func(sentence_pairs)
        ranking = {pid: score for pid, score in zip(range(len(snippets)), scores)}
        ranking = dict(sorted(ranking.items(), key=lambda item: item[1], reverse=True))
        new_snippets = []
        new_scores = []
        for pid, score in ranking.items():
            new_snippets.append(snippets[pid])
            if mode == "score":
                new_scores.append(score)
            else:
                new_scores.append(1 if pid<5 else 0)
        return new_snippets, new_scores

    def get_prompt(self, data):
        retrieved_snippets = data["snippets"]
        contexts = ["Document [{:d}]: {:s}".format(idx, retrieved_snippets[idx]["contents"].replace("\n", " ")) for idx in
                    range(len(retrieved_snippets))]
        if len(contexts) == 0:
            contexts = [""]
        contexts = [self.reader_tokenizer.decode(
            self.reader_tokenizer.encode("\n".join(contexts), add_special_tokens=False)[:self.context_length])]
        options = "".join([k + ": " + v + "\n" for k, v in data["options"].items()])
        prompt = self.templates["rag"].format(documents=contexts[0], question=data["question"]+"\n"+options)
        return prompt

    def vllm_answer(self, batch_data):
        self.reader_sampling_params = SamplingParams(temperature=0.1, seed=42, max_tokens=512)
        self.reader_model = LLM(model=self.reader_name, tensor_parallel_size=2, disable_custom_all_reduce=True,
                         trust_remote_code=True, gpu_memory_utilization=0.9)
        print(f">>>>>>Reader have downloaded from {self.reader_name}!")
        results = {}
        batch_size = 128
        batchs = [batch_data[i:i + batch_size] for i in range(0, len(batch_data), batch_size)]
        print(f"一共{len(batch_data)}个query！")
        global_num = 0
        global_num += 1
        print_flag1, print_flag2 = 0, 0
        for batch in tqdm(batchs):
            batch_text = []
            for entry in batch:
                input_prompt = self.get_prompt(entry)
                if print_flag1 == 1 and global_num == 1:
                    print()
                    print("test prompt")
                    print(input_prompt)
                print_flag1 += 1
                sys_prompt = "You are a helpful assistant."
                messages = [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": input_prompt}
                ]
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                batch_text.append(text)
            outputs = self.reader_model.generate(batch_text, self.reader_sampling_params, use_tqdm=False)
            for entry, output in zip(batch, outputs):
                response = output.outputs[0].text
                generated_text = re.sub("\s+", " ", response)
                if print_flag2 == 1 and global_num == 1:
                    print()
                    print("generated_text")
                    print(generated_text)
                print_flag2 += 1
                results[entry["id"]] = generated_text
        return results