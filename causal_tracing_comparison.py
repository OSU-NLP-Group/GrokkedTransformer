import numpy as np
import json, jsonlines
import matplotlib.pyplot as plt
from eval_qa import eval_file, eval_items
import os
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer, GPTNeoXForCausalLM, GPTNeoXTokenizerFast, GPTNeoXConfig
from tqdm import tqdm
import torch
import torch.nn.functional as F
import pandas as pd
from copy import deepcopy
import random
import argparse

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", default=None, type=str, required=True, help="dataset name")
    parser.add_argument("--model_dir", default=None, type=str, help="parent directory of saved model checkpoints")
    parser.add_argument("--save_path", default=None, type=str, help="path to save result")

    parser.add_argument("--num_layer", default=8, type=int, help="number of layer of the model")
    parser.add_argument("--wd", default=0.1, type=float, help="weight decay being used")

    args = parser.parse_args()
    dataset, model_dir = args.dataset, args.model_dir

    directory = os.path.join(model_dir, "{}_{}_{}".format(dataset, args.wd, args.num_layer))

    device = torch.device('cuda:7')  

    all_atomic = set()
    atomic_dict = dict()
    with open("data/{}/train.json".format(dataset)) as f:
        train_items = json.load(f)
    for item in tqdm(train_items):
        temp = item['target_text'].strip("><").split("><")
        if len(temp) != 4:
            continue
        h,r,t = temp[:3]
        atomic_dict[(h,r)] = t
        all_atomic.add((h,r,t))

    train_atomic = set()
    for item in tqdm(train_items):
        temp = item['target_text'].strip("><").split("><")
        if len(temp) == 4:
            continue
        r, e1, e2 = temp[0], temp[2], temp[4]
        val1 = atomic_dict[(e1, r)]
        val2 = atomic_dict[(e2, r)]
        train_atomic.add((e1, r, val1))
        train_atomic.add((e2, r, val2))

    test_atomic = all_atomic - train_atomic
    print(len(train_atomic), len(test_atomic))

    r2ht_train = dict()
    for (h,r,t) in train_atomic:
        if r not in r2ht_train:
            r2ht_train[r] = []
        r2ht_train[r].append((h,t))

    def return_rank(hd, word_embedding_, token, metric='dot', token_list=None, position=None):
        if metric == 'dot':
            word_embedding = word_embedding_
        elif metric == 'cos':
            word_embedding = F.normalize(word_embedding_, p=2, dim=1)
        else:
            assert False

        logits_ = torch.matmul(hd, word_embedding.T)

        if position is None:
            rank = []
            for j in range(len(logits_)):
                log = logits_[j].cpu().numpy()
                if token_list is None:
                    temp = [[i, log[i]] for i in range(len(log))]
                else:
                    temp = [[i, log[i]] for i in token_list]
                temp.sort(key=lambda var: var[1], reverse=True)
                rank.append([var[0] for var in temp].index(token))
            return rank
        
        j = position
        log = logits_[j]
        log = log.cpu().numpy()
        if token_list is None:
            temp = [[i, log[i]] for i in range(len(log))]
        else:
            temp = [[i, log[i]] for i in token_list]
        temp.sort(key=lambda var: var[1], reverse=True)
        return [var[0] for var in temp].index(token)

    with open("data/{}/test.json".format(dataset)) as f:
        pred_data = json.load(f)
    d = dict()
    for item in pred_data:
        t = item['type']
        if t not in d:
            d[t] = []
        d[t].append(item)

    all_checkpoints = [checkpoint for checkpoint in os.listdir(directory) if checkpoint.startswith("checkpoint")]
    all_checkpoints.sort(key=lambda var: int(var.split("-")[1]))

    results = []

    np.random.seed(0)
    split = 'train_inferred'
    rand_inds = np.random.choice(len(d[split]), 300, replace=False).tolist()

    for checkpoint in tqdm(all_checkpoints):
        print("now checkpoint", checkpoint)
        model_path = os.path.join(directory, checkpoint)
        model = GPT2LMHeadModel.from_pretrained(model_path).to(device)
        word_embedding = model.lm_head.weight.data
        tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        tokenizer.padding_side = "left" 
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = model.config.eos_token_id

        full_list = []
        for index in tqdm(rand_inds):

            random.seed(index)
            
            res_dict = dict()

            temp = d[split][index]['target_text'].strip("><").split("><")
            r, e1, e2, t = temp[0], temp[2], temp[4], temp[5]
            val_1, val_2 = atomic_dict[(e1, r)], atomic_dict[(e2, r)]

            query = "<{}><q><{}><mask><{}>".format(r, e1, e2)
            decoder_temp = tokenizer([query], return_tensors="pt", padding=True)
            decoder_input_ids, decoder_attention_mask = decoder_temp["input_ids"], decoder_temp["attention_mask"]
            decoder_input_ids, decoder_attention_mask = decoder_input_ids.to(device), decoder_attention_mask.to(device)
            
            with torch.no_grad():
                outputs = model(
                    input_ids=decoder_input_ids,
                    attention_mask=decoder_attention_mask,
                    output_hidden_states=True
                )
            all_hidden_states = outputs['hidden_states']

            rank_before = return_rank(all_hidden_states[8][0, :, :], word_embedding, tokenizer("<"+t+">")['input_ids'][0], position=-1)
            res_dict['rank_before'] = rank_before

            # MRRs
            for layer_to_intervene in range(1, 8):
                hidden_states_orig = all_hidden_states[layer_to_intervene]
                with torch.no_grad():
                    temp = model.transformer.ln_f(hidden_states_orig)

                val1_rank_pos2 = return_rank(temp[0, :, :], word_embedding, tokenizer("<"+val_1+">")['input_ids'][0], position=2)
                val2_rank_pos4 = return_rank(temp[0, :, :], word_embedding, tokenizer("<"+val_2+">")['input_ids'][0], position=4)

                label0_rank = return_rank(temp[0, :, :], word_embedding, tokenizer("<{}_0>".format(r))['input_ids'][0], position=0)
                label1_rank = return_rank(temp[0, :, :], word_embedding, tokenizer("<{}_1>".format(r))['input_ids'][0], position=0)
                label2_rank = return_rank(temp[0, :, :], word_embedding, tokenizer("<{}_2>".format(r))['input_ids'][0], position=0)

                res_dict['val1_rank_pos2_'+str(layer_to_intervene)] = val1_rank_pos2
                res_dict['val2_rank_pos4_'+str(layer_to_intervene)] = val2_rank_pos4

                res_dict['label0_rank_pos0_'+str(layer_to_intervene)] = label0_rank
                res_dict['label1_rank_pos0_'+str(layer_to_intervene)] = label1_rank
                res_dict['label2_rank_pos0_'+str(layer_to_intervene)] = label2_rank
            
            # perturb the 1st entity
            val_1_int, val_2_int = int(val_1), int(val_2)
            if val_1_int == val_2_int:
                closest = {val_2_int-1, val_2_int+1}
            else:
                closest = {val_2_int}

            all_ = set()
            for (entity, value) in r2ht_train[r]:
                if entity != e1 and int(value) in closest:
                    all_.add(entity)
            query = "<{}><q><{}><mask><{}>".format(r, random.choice(list(all_)), e2)

            decoder_temp = tokenizer([query], return_tensors="pt", padding=True)
            decoder_input_ids_, decoder_attention_mask = decoder_temp["input_ids"], decoder_temp["attention_mask"]
            decoder_input_ids_, decoder_attention_mask = decoder_input_ids_.to(device), decoder_attention_mask.to(device)

            with torch.no_grad():
                outputs_ctft = model(
                    input_ids=decoder_input_ids_,
                    attention_mask=decoder_attention_mask,
                    output_hidden_states=True
                )
            all_hidden_states_ctft = outputs_ctft['hidden_states']

            for layer_to_intervene in range(1, 8):
                hidden_states = all_hidden_states[layer_to_intervene].clone()
                hidden_states_ctft = all_hidden_states_ctft[layer_to_intervene]
                # intervene
                hidden_states[0, 2, :] = hidden_states_ctft[0, 2, :]
                hidden_states[0, 3, :] = hidden_states_ctft[0, 3, :]

                with torch.no_grad():
                    for i in range(layer_to_intervene, 8):
                        f_layer = model.transformer.h[i]
                        # attn
                        residual = hidden_states
                        hidden_states = f_layer.ln_1(hidden_states)
                        attn_output = f_layer.attn(hidden_states)[0] 
                        hidden_states = attn_output + residual
                        # mlp
                        residual = hidden_states
                        hidden_states = f_layer.ln_2(hidden_states)
                        feed_forward_hidden_states = f_layer.mlp.c_proj(f_layer.mlp.act(f_layer.mlp.c_fc(hidden_states)))
                        hidden_states = residual + feed_forward_hidden_states
                    # final ln
                    hidden_states = model.transformer.ln_f(hidden_states)
                # print("--------")
                rank_after = return_rank(hidden_states[0, :, :], word_embedding, tokenizer("<"+t+">")['input_ids'][0], position=-1)
                res_dict['e1_'+str(layer_to_intervene)] = rank_after

            # perturb the 2nd entity
            val_1_int, val_2_int = int(val_1), int(val_2)
            if val_1_int == val_2_int:
                closest = {val_1_int-1, val_1_int+1}
            else:
                closest = {val_1_int}

            all_ = set()
            ht_list = r2ht_train[r]
            for (entity, value) in ht_list:
                if entity != e2 and int(value) in closest:
                    all_.add(entity)
            query = "<{}><q><{}><mask><{}>".format(r, e1, random.choice(list(all_)))

            decoder_temp = tokenizer([query], return_tensors="pt", padding=True)
            decoder_input_ids_, decoder_attention_mask = decoder_temp["input_ids"], decoder_temp["attention_mask"]
            decoder_input_ids_, decoder_attention_mask = decoder_input_ids_.to(device), decoder_attention_mask.to(device)

            with torch.no_grad():
                outputs_ctft = model(
                    input_ids=decoder_input_ids_,
                    attention_mask=decoder_attention_mask,
                    output_hidden_states=True
                )
            all_hidden_states_ctft = outputs_ctft['hidden_states']

            for layer_to_intervene in range(1, 8):
                hidden_states = all_hidden_states[layer_to_intervene].clone()
                hidden_states_ctft = all_hidden_states_ctft[layer_to_intervene]
                # intervene
                hidden_states[0, 4, :] = hidden_states_ctft[0, 4, :]

                with torch.no_grad():
                    for i in range(layer_to_intervene, 8):
                        f_layer = model.transformer.h[i]
                        # attn
                        residual = hidden_states
                        hidden_states = f_layer.ln_1(hidden_states)
                        attn_output = f_layer.attn(hidden_states)[0] 
                        hidden_states = attn_output + residual
                        # mlp
                        residual = hidden_states
                        hidden_states = f_layer.ln_2(hidden_states)
                        feed_forward_hidden_states = f_layer.mlp.c_proj(f_layer.mlp.act(f_layer.mlp.c_fc(hidden_states)))
                        hidden_states = residual + feed_forward_hidden_states
                    # final ln
                    hidden_states = model.transformer.ln_f(hidden_states)
                # print("--------")
                rank_after = return_rank(hidden_states[0, :, :], word_embedding, tokenizer("<"+t+">")['input_ids'][0], position=-1)
                res_dict['e2_'+str(layer_to_intervene)] = rank_after
            
            # intervene on the attr
            all_ = set()
            for attr in r2ht_train.keys():
                if attr != r:
                    all_.add(attr)
            query = "<{}><q><{}><mask><{}>".format(random.choice(list(all_)), e1, e2)

            decoder_temp = tokenizer([query], return_tensors="pt", padding=True)
            decoder_input_ids_, decoder_attention_mask = decoder_temp["input_ids"], decoder_temp["attention_mask"]
            decoder_input_ids_, decoder_attention_mask = decoder_input_ids_.to(device), decoder_attention_mask.to(device)

            with torch.no_grad():
                outputs_ctft = model(
                    input_ids=decoder_input_ids_,
                    attention_mask=decoder_attention_mask,
                    output_hidden_states=True
                )
            all_hidden_states_ctft = outputs_ctft['hidden_states']

            for layer_to_intervene in range(1, 8):
                hidden_states = all_hidden_states[layer_to_intervene].clone()
                hidden_states_ctft = all_hidden_states_ctft[layer_to_intervene]
                # intervene
                hidden_states[0, 0, :] = hidden_states_ctft[0, 0, :]
                hidden_states[0, 1, :] = hidden_states_ctft[0, 1, :]

                with torch.no_grad():
                    for i in range(layer_to_intervene, 8):
                        f_layer = model.transformer.h[i]
                        # attn
                        residual = hidden_states
                        hidden_states = f_layer.ln_1(hidden_states)
                        attn_output = f_layer.attn(hidden_states)[0] 
                        hidden_states = attn_output + residual
                        # mlp
                        residual = hidden_states
                        hidden_states = f_layer.ln_2(hidden_states)
                        feed_forward_hidden_states = f_layer.mlp.c_proj(f_layer.mlp.act(f_layer.mlp.c_fc(hidden_states)))
                        hidden_states = residual + feed_forward_hidden_states
                    # final ln
                    hidden_states = model.transformer.ln_f(hidden_states)
                # print("--------")
                rank_after = return_rank(hidden_states[0, :, :], word_embedding, tokenizer("<"+t+">")['input_ids'][0], position=-1)
                res_dict['a_'+str(layer_to_intervene)] = rank_after

            # print(res_dict)
            full_list.append(res_dict)

        results.append(full_list)

    with open(args.save_path, "w", encoding='utf-8') as f:
        json.dump(results, f)

if __name__ == '__main__':
    main()