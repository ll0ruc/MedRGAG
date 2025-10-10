import os
import re
import json
import numpy as np

class QADataset:

    def __init__(self, data, dir="./benchmark.json"):
        self.data = data.lower().split("_")[0]
        benchmark = json.load(open(dir))
        if self.data not in benchmark:
            raise KeyError("{:s} not supported".format(data))
        self.dataset = benchmark[self.data]
        self.index = sorted(self.dataset.keys())

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, key):
        if type(key) == int:
            return [self.index[key], self.dataset[self.index[key]]]
        elif type(key) == slice:
            return [self.__getitem__(i) for i in range(self.__len__())[key]]
        else:
            raise KeyError("Key type not supported.")

def get_acc(results):
    flag = False
    pred = []
    empty_count = 0
    na_count = 0
    answer_list = ["A", "B", "C", "D"]
    answer2idx = {ans: i for i, ans in enumerate(answer_list)}
    # for i, fpath in enumerate(sorted([f for f in os.listdir(save_dir) if f.endswith(".json")])[:total_len]):
    for result in results:
        model_answer = result["model_answer"]
        if model_answer in answer_list:
            pred.append(answer_list.index(model_answer))
        else:
            pred.append(-1)

    truth = [answer2idx[item['answer']] for item in results]
    if len(pred) < len(truth):
        truth = truth[:len(pred)]
        flag = True

    acc = (np.array(truth) == np.array(pred)).mean()
    std = np.sqrt(acc * (1 - acc) / len(truth))
    return acc, std, flag

def locate_answer(sentence:str):

    ans = re.findall("^\s*(A|B|C|D)$", sentence)
    if len(ans) > 0:
        return ans[0].upper()
    
    ans = re.findall("^\s*(A|B|C|D) or", sentence)
    if len(ans) > 0:
        return ans[0].upper()
    
    ans = re.findall("^\s*(A|B|C|D) and", sentence)
    if len(ans) > 0:
        return ans[0].upper()
        
    ans = re.findall("^\s*(A|B|C|D)/", sentence)
    if len(ans) > 0:
        return ans[0].upper()
    
    ans = re.findall("^\s*(A|B|C|D),", sentence)
    if len(ans) > 0:
        return ans[0].upper()
    
    ans = re.findall("[Oo]ption (A|B|C|D)", sentence)
    if len(ans) > 0:
        return ans[0]

    ans = re.findall(":\s*(A|B|C|D)", sentence)
    if len(ans) > 0:
        return ans[0].upper()

    ans = re.findall("^\s*(A|B|C|D)\.", sentence)
    if len(ans) > 0:
        return ans[0].upper()

    ans = re.findall("^\s*(A|B|C|D)\"", sentence)
    if len(ans) > 0:
        return ans[0].upper()
    
    ans = re.findall("^\s*(A|B|C|D):", sentence)
    if len(ans) > 0:
        return ans[0].upper()

    return "A"
