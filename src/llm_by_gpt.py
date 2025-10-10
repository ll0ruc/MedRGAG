# %%
import os
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
import argparse
import random
from template import knowledge_summary_template, knowledge_explore_template, llm_selection_prompt
from evaluate_utils import QADataset
import re
import sys
from collections import defaultdict

class GPT:
    def __init__(self, model_name='gpt-4o-mini-2024-07-18') -> None:
        self.max_wrong_time = 2
        self.init_client()
        self.model_name = model_name
        print(f'use model of {self.model_name}')
        self.temperature = 1.2
        print(f'use temperature is {self.temperature}')

    def init_client(self):
        self.client = OpenAI(
            api_key="sk-***********************",
            base_url="https://*******",
            max_retries=self.max_wrong_time
        )

    def call(self, content):
        chat_completion = self.client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": content}
            ],
            model=self.model_name,
            temperature=self.temperature,
        )
        response = chat_completion.choices[0].message.content
        return response


def read_data(dataset_name, mode):
    datasets = {dataset_name: QADataset(dataset_name)}
    if mode == "summary":
        with open(f"./outputs/{dataset_name}/docs/retrieval/snippets.json", "r", encoding="utf-8") as f:
            retrieved_snippets = json.load(f)
        entry_data = []
        for id, data in tqdm(datasets[dataset_name][:]):  ## 10178//2
            question = data["question"]
            ret_contents = retrieved_snippets[str(id)]["snippets"][:5]
            options = "".join([k + ": " + v + "\n" for k, v in data["options"].items()]).strip()
            contexts = ["Document: {:s}".format(ret_contents[idx]["contents"].replace("\n", " ")) for
                        idx in range(len(ret_contents))]
            for num, context in enumerate(contexts):
                entry_data.append({
                    "id": id,
                    "ret_id": num,
                    "question": question,
                    "options": options,
                    "ret_content": context,
                })
    elif mode == "explore":
        with open(f"./outputs/{dataset_name}/docs/generate/snippets_inter_summary.json", "r", encoding="utf-8") as f:
            retrieved_snippets = json.load(f)
        entry_data = []
        for id, data in tqdm(datasets[dataset_name][:]):  ## 10178//2
            if id not in retrieved_snippets:
                continue
            question = data["question"]
            options = "".join([k + ": " + v + "\n" for k, v in data["options"].items()]).strip()
            sum_data = retrieved_snippets[id]
            valid_contexts = [context for context in sum_data["summary_snippets"] if "No useful information" not in context]
            if len(valid_contexts) == 0:
                contexts = "No useful information provided."
            else:
                contexts = "\n".join(["Information [{:d}]: {:s}".format(idx, valid_contexts[idx].replace("\n", " ")) for
                            idx in range(len(valid_contexts))])
            entry_data.append({
                "id": id,
                "question": question,
                "options": options,
                "summary_snippets": sum_data["summary_snippets"],
                "sum_content": contexts,
            })
    else:
        ret_path = f"./outputs/{dataset_name}/docs/retrieval/snippets.json"
        with open(ret_path, "r", encoding="utf-8") as f:
            retrieved_snippets = json.load(f)
        gen_path = f"./outputs/{dataset_name}/docs/generate/snippets_gen_gpt.json"
        with open(gen_path, "r", encoding="utf-8") as f:
            generated_snippets = json.load(f)
        entry_data = []
        for id, data in tqdm(datasets[dataset_name][:]):  ## 10178//2
            question = data["question"]
            options = "".join([k + ": " + v + "\n" for k, v in data["options"].items()]).strip()
            ret_snippets = retrieved_snippets[id]["snippets"][:5]
            gen_snippets = generated_snippets[id]["snippets"][:5]
            snippets = ret_snippets + gen_snippets
            contexts = "\n".join(
                ["[{:d}] {:s}".format(idx, snippets[idx]["contents"].replace("\n", " ")) for idx in
                 range(len(snippets))])
            entry_data.append({
                "id": id,
                "question": question,
                "options": options,
                "snippets": contexts,
            })
    return entry_data

wrongtime = 0


def extract_knowledge_str(text):
    point1 = re.search(r'Knowledge 1:\s*(.+)', text)
    point2 = re.search(r'Knowledge 2:\s*(.+)', text)
    point3 = re.search(r'Knowledge 3:\s*(.+)', text)
    kpoint1 = point1.group(1).strip() if point1 else "None"
    kpoint2 = point2.group(1).strip() if point2 else "None"
    kpoint3 = point3.group(1).strip() if point3 else "None"
    if kpoint1 == "None" or kpoint2 == "None" or kpoint3 == "None":
        # pass
        print(f">>Knowledge Point Extract Error in {text}")
    return [kpoint1, kpoint2, kpoint3]

def extract_ids(llm_output):
    match = re.findall(r"Final Selection\s*(.*?)(?=Final Selection|$)", llm_output, flags=re.DOTALL)
    if not match:
        print("Error parsing ids from LLM output:", llm_output)
        return []
    ids = re.findall(r"\[(\d+)\]", match[-1])
    ids = [int(i)%10 for i in ids]
    return ids[:5]

def extract_knowledge_for_text(d, gpt, writer, mode, data_path):
    if mode == "summary":
        chatgpt_query = knowledge_summary_template.format(question=d["question"] + "\n" + d["options"], documents=d["ret_content"])
        output = gpt.call(chatgpt_query)
        new_data = {
            "id": d["id"],
            "ret_id": d["ret_id"],
            "summary_snippets": output,
            "llm_output": output,
        }
    elif mode == "explore":
        if data_path in ["medmcqa", "bioasq"]:
            knowledge_template = knowledge_explore_template_two_know
        else:
            knowledge_template = knowledge_explore_template
        chatgpt_query = knowledge_template.format(question=d["question"] + "\n" + d["options"], documents=d["sum_content"])
        output = gpt.call(chatgpt_query)
        generated_knowledge = extract_knowledge_str(output)
        new_data = {
            "id": d["id"],
            "summary_snippets": d["summary_snippets"],
            "explore_snippets": generated_knowledge,
            "llm_output": output,
                    }
    else:
        chatgpt_query = llm_selection_prompt.format(question=d["question"] + "\n" + d["options"],
                                                             documents=d["snippets"])
        output = gpt.call(chatgpt_query)
        final_list = extract_ids(output)
        new_data = {
            "id": d["id"],
            "final_list": final_list,
            "llm_output": output,
        }
    writer.write(json.dumps(new_data, ensure_ascii=False) + "\n")

    return 1

def merge_knowledge(data_path, mode):
    if mode =="summary":
        data = {}
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                e = json.loads(line)
                id = e["id"]
                ret_id = e["ret_id"]
                summary = e["summary_snippets"]
                if id not in data:
                    data[id] = [0] * 5
                data[id][ret_id] = summary
        save_data = {}
        for k, v in data.items():
            item = {
                "id": k,
                "summary_snippets": v,
                "explore_snippets": "",
            }
            save_data[k] = item
        path = data_path.replace(".jsonl", ".json")
        with open(path, 'w') as f:
            json.dump(save_data, f, indent=4)
    elif mode == "explore":
        data = defaultdict(dict)
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                e = json.loads(line)
                id = e["id"]
                summary = e["summary_snippets"]
                explore = e["explore_snippets"]
                data[id]["summary_snippets"] = summary
                data[id]["explore_snippets"] = explore
        save_data = {}
        for k, v in data.items():
            item = {
                "id": k,
                "summary_snippets": v["summary_snippets"],
                "explore_snippets": v["explore_snippets"],
            }
            save_data[k] = item
        path = data_path.replace(".jsonl", ".json")
        with open(path, 'w') as f:
            json.dump(save_data, f, indent=4)
    else:
        data = defaultdict(dict)
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                e = json.loads(line)
                id = e["id"]
                final_list = e["final_list"]
                data[id]["final_list"] = final_list
        save_data = {}
        for k, v in data.items():
            item = {
                "id": k,
                "final_list": v["final_list"],
            }
            save_data[k] = item
        path = data_path.replace(".jsonl", ".json")
        with open(path, 'w') as f:
            json.dump(save_data, f, indent=4)
    return save_data

def Function_LLM(mode="summary", data_name="medqa", model_name="gpt-4o-mini-2024-07-18"):
    mode = args.mode
    data = read_data(data_name, mode)[:]
    print(f"read data:{len(data)}")
    gpt = GPT(model_name)
    dir_path = f"./outputs/{data_name}/docs/generate/"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    if mode == "summary":
        save_path = dir_path + "snippets_inter_summary.jsonl"
    elif mode == "explore":
        save_path = dir_path + "snippets_inter_explore.jsonl"
    else:
        save_path = dir_path + "snippets_final_select.jsonl"

    writer = open(save_path, mode="w", encoding="utf-8")
    with ThreadPoolExecutor(max_workers=16) as executor:
        results = list(tqdm(executor.map(lambda x: extract_knowledge_for_text(x, gpt, writer, mode, args.data_path), data), total=len(data),
                            desc="Processing samples", unit="sample"))

    writer.close()
    print(f'finish_')
    output_data = merge_knowledge(save_path, mode)
    return output_data