from sentence_transformers.models import Transformer, Pooling
from sentence_transformers import SentenceTransformer
import os
import faiss
import json
import torch
import tqdm
import numpy as np

corpus_names = {
    "Textbooks": ["textbooks"],
    "Wikipedia": ["wikipedia"],
    "MedText": ["textbooks", "wikipedia"],
}

retriever_names = {
    "BM25": ["bm25"]
}

class Retriever:
    def __init__(self, retriever_name="bm25", corpus_name="MedText", db_dir="./corpus", HNSW=False, **kwarg):
        self.retriever_name = retriever_name
        self.corpus_name = corpus_name
        self.db_dir = db_dir
        if not os.path.exists(self.db_dir):
            os.makedirs(self.db_dir)
        self.chunk_dir = os.path.join(self.db_dir, self.corpus_name, "chunk")
        if not os.path.exists(self.chunk_dir):
            print("Cloning the {:s} corpus from Huggingface...".format(self.corpus_name))
            os.system("git clone https://huggingface.co/datasets/MedRAG/{:s} {:s}".format(corpus_name, os.path.join(self.db_dir, self.corpus_name)))
        self.index_dir = os.path.join(self.db_dir, self.corpus_name, "index", self.retriever_name.split("/")[-1].replace("Query-Encoder", "Article-Encoder"))
        from pyserini.search.lucene import LuceneSearcher
        self.metadatas = None
        self.embedding_function = None
        if os.path.exists(self.index_dir):
            self.index = LuceneSearcher(os.path.join(self.index_dir))
        else:
            os.system("python -m pyserini.index.lucene --collection JsonCollection --input {:s} --index {:s} --generator DefaultLuceneDocumentGenerator --threads 16".format(self.chunk_dir, self.index_dir))
            self.index = LuceneSearcher(os.path.join(self.index_dir))

    def get_relevant_documents(self, question, k=32, id_only=False, **kwarg):
        assert type(question) == str
        question = [question]
        res_ = [[]]
        hits = self.index.search(question[0], k=k)
        res_[0].append(np.array([h.score for h in hits]))
        ids = [h.docid for h in hits]
        indices = [{"source": '_'.join(h.docid.split('_')[:-1]), "index": eval(h.docid.split('_')[-1])} for h in hits]
        scores = res_[0][0].tolist()
        if id_only:
            return [{"id":i} for i in ids], scores
        else:
            return self.idx2txt(indices), scores

    def idx2txt(self, indices): # return List of Dict of str
        '''
        Input: List of Dict( {"source": str, "index": int} )
        Output: List of str
        '''
        return [json.loads(open(os.path.join(self.chunk_dir, i["source"]+".jsonl")).read().strip().split('\n')[i["index"]]) for i in indices]

class RetrievalSystem:
    def __init__(self, retriever_name="BM25", corpus_name="MedText", db_dir="./corpus", HNSW=False, cache=False):
        self.retriever_name = retriever_name
        self.corpus_name = corpus_name
        assert self.corpus_name in corpus_names
        assert self.retriever_name in retriever_names
        self.retrievers = []
        for corpus in corpus_names[self.corpus_name]:
            self.retrievers.append(Retriever(retriever_names[self.retriever_name][0], corpus, db_dir, HNSW=HNSW))
        self.cache = cache
        if self.cache:
            self.docExt = DocExtracter(cache=True, corpus_name=self.corpus_name, db_dir=db_dir)
        else:
            self.docExt = None
    
    def retrieve(self, question, k=32, id_only=False):
        assert type(question) == str
        if self.cache:
            id_only = True
        texts, scores = [], []
        for i in range(len(corpus_names[self.corpus_name])):
            t, s = self.retrievers[i].get_relevant_documents(question, k=k, id_only=id_only)
            texts.extend(t)
            scores.extend(s)
        if self.cache:
            texts = self.docExt.extract(texts)
        return texts, scores


class DocExtracter:
    def __init__(self, db_dir="./corpus", cache=False, corpus_name="MedText"):
        self.db_dir = db_dir
        self.cache = cache
        print("Initializing the document extracter...")
        for corpus in corpus_names[corpus_name]:
            if not os.path.exists(os.path.join(self.db_dir, corpus, "chunk")):
                print("Cloning the {:s} corpus from Huggingface...".format(corpus))
                os.system("git clone https://huggingface.co/datasets/MedRAG/{:s} {:s}".format(corpus, os.path.join(self.db_dir, corpus)))
        if self.cache:
            if os.path.exists(os.path.join(self.db_dir, "_".join([corpus_name, "id2text.json"]))):
                self.dict = json.load(open(os.path.join(self.db_dir, "_".join([corpus_name, "id2text.json"]))))
            else:
                self.dict = {}
                for corpus in corpus_names[corpus_name]:
                    for fname in tqdm.tqdm(sorted(os.listdir(os.path.join(self.db_dir, corpus, "chunk")))):
                        if open(os.path.join(self.db_dir, corpus, "chunk", fname)).read().strip() == "":
                            continue
                        for i, line in enumerate(open(os.path.join(self.db_dir, corpus, "chunk", fname)).read().strip().split('\n')):
                            item = json.loads(line)
                            _ = item.pop("contents", None)
                            # assert item["id"] not in self.dict
                            self.dict[item["id"]] = item
                with open(os.path.join(self.db_dir, "_".join([corpus_name, "id2text.json"])), 'w') as f:
                    json.dump(self.dict, f)
        else:
            if os.path.exists(os.path.join(self.db_dir, "_".join([corpus_name, "id2path.json"]))):
                self.dict = json.load(open(os.path.join(self.db_dir, "_".join([corpus_name, "id2path.json"]))))
            else:
                self.dict = {}
                for corpus in corpus_names[corpus_name]:
                    for fname in tqdm.tqdm(sorted(os.listdir(os.path.join(self.db_dir, corpus, "chunk")))):
                        if open(os.path.join(self.db_dir, corpus, "chunk", fname)).read().strip() == "":
                            continue
                        for i, line in enumerate(open(os.path.join(self.db_dir, corpus, "chunk", fname)).read().strip().split('\n')):
                            item = json.loads(line)
                            # assert item["id"] not in self.dict
                            self.dict[item["id"]] = {"fpath": os.path.join(corpus, "chunk", fname), "index": i}
                with open(os.path.join(self.db_dir, "_".join([corpus_name, "id2path.json"])), 'w') as f:
                    json.dump(self.dict, f, indent=4)
        print("Initialization finished!")

    def extract(self, ids):
        if self.cache:
            output = []
            for i in ids:
                item = self.dict[i] if type(i) == str else self.dict[i["id"]]
                output.append(item)
        else:
            output = []
            for i in ids:
                item = self.dict[i] if type(i) == str else self.dict[i["id"]]
                output.append(json.loads(open(os.path.join(self.db_dir, item["fpath"])).read().strip().split('\n')[item["index"]]))
        return output

def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        all_data = json.load(f)
    return all_data