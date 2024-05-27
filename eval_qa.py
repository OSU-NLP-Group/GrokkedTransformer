import logging
import argparse
import os
import json
from tqdm import tqdm

def eval_res(a, b):
    assert b.count("</a>") in [0,1]
    if b.count("</a>") == 0:
        return int(a.startswith(b))
    b = b.split("</a>")[0]
    a = a.split("</a>")[0]
    if b.count("<a>") == 1:
        # extract and compare the part between <a> and </a>
        if a.count("<a>") != 1:
            return 0
        a = a.split("<a>")[1]
        b = b.split("<a>")[1]
        return int(a==b)
    if b.count("<a>") == 0:
        return int(a==b)
    assert False

def parse_target(target_text):
    temp = target_text.split("</a>")[0]
    q, a = temp.split("<a>")
    attr, q = q.split("<q>")
    q = q.split("</q>")[0]
    # print(attr, q)
    attr = attr.strip("><")
    h, _, t = q.strip("><").split("><")
    return attr, h, t


def eval_items(all_items, partition_atomic=False, test_entities=None):
    acc = dict()   # maps each type of example to the corresponding list of eval results
    for item in all_items:
        if 'type' not in item:
            t = 'test_inferred'
        else:
            t = item['type']
        
        if "model_output" in item:
            pred, gold = item["model_output"], item["target_text"]
        else:
            pred, gold = item["model output"], item["target text"]

        if t == 'train_atomic' and partition_atomic:
            head, rel, _ = gold.split("<a>")[0].strip("><").split("><")[1:-1]
            if rel in test_entities:
                # determine whether it's train or test atomic fact
                if head in test_entities[rel]:
                    t = "test_atomic"
                else:
                    t = "train_atomic"

        if t not in acc:
            acc[t] = []
        acc[t].append(eval_res(pred, gold))
    return acc


def eval_file(dir_, fn='all_items.json', partition_atomic=False):
    
    test_entities = dict()
    if partition_atomic:
        with open("data/{}/valid.json".format("_".join(dir_.split("/")[-2].split("_")[:2]))) as f:
            valid = json.load(f)
        for item in valid:
            # print(item['target_text'])
            attr, h, t = parse_target(item['target_text'])
            if attr.endswith("_q"):
                attr = attr.replace("_q", "")
            # print(attr, h, t)
            if attr not in test_entities:
                test_entities[attr] = set()
            test_entities[attr] = test_entities[attr] | {h, t}

    scores_dict = dict()

    for folder_name in tqdm(os.listdir(dir_)):
        if not folder_name.startswith("checkpoint"):
            continue
        
        if fn not in os.listdir(os.path.join(dir_, folder_name)):
            continue
        
        with open(os.path.join(dir_, folder_name, fn)) as f:
            all_items = json.load(f)

        acc = eval_items(all_items, partition_atomic=partition_atomic, test_entities=test_entities)
                
        scores_dict[folder_name] = [(t, round(sum(acc[t])/len(acc[t]), 3)) for t in acc]

    # sort via checkpoint step. all folder name are in format "checkpoint-<step>-*"
    temp = []
    for folder_name in scores_dict:
        temp.append((folder_name, scores_dict[folder_name]))
    temp.sort(key=lambda var: int(var[0].split("-")[1]))
    
    return temp
    
def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=None, type=str, required=True, help="Input file dir.")
    parser.add_argument("--fn", default='all_items.json', type=str, help="")
    parser.add_argument("--partition_atomic", action="store_true", help="")
    args = parser.parse_args()

    scores_dict = eval_file(args.dir, args.fn, args.partition_atomic)
    temp = []
    for (folder_name, val) in scores_dict:
        temp.append((folder_name, "; ".join(["{}: {}".format(t, res) for (t, res) in val])))

    for (folder_name, res) in temp:
        print(folder_name, "|", res)
    

if __name__ == '__main__':
    main()

