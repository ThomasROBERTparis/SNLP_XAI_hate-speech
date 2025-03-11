import pandas as pd
import numpy as np
import os
import pickle
import re
from collections import Counter
from tqdm import tqdm

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config

import ilm.ilm.tokenize_util
from ilm.ilm.infer import infill_with_ilm
from perturbation_functions import calculate_necc_and_suff, gen_num_samples_table, gen_probs_table

import contextlib
import io


MODEL_DIR = './ilm/Models/ILM/'

# tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")



tokenizer = ilm.ilm.tokenize_util.Tokenizer.GPT2

with open(os.path.join(MODEL_DIR, 'additional_ids_to_tokens.pkl'), 'rb') as f:
    additional_ids_to_tokens = pickle.load(f)
    print(additional_ids_to_tokens)
additional_tokens_to_ids = {v:k for k, v in additional_ids_to_tokens.items()}
try:
    ilm.ilm.tokenize_util.update_tokenizer(additional_ids_to_tokens, tokenizer)
except ValueError:
    print('Already updated')
print(additional_tokens_to_ids)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

vocab_size = ilm.ilm.tokenize_util.vocab_size(tokenizer=tokenizer)

model.resize_token_embeddings(vocab_size)

model.eval()
_ = model.to(device)

num_samples = gen_num_samples_table(20, 100)
probs_table = gen_probs_table(20)
mask_tokn = additional_tokens_to_ids['<|infill_ngram|>']

orig_texts = []
necc_perturbed = []
suff_perturbed = []
necc_masks = []
suff_masks = []

with open("Data/HateCheck_test_suite_cases.txt", "r") as ff:
    with tqdm(total=240) as pbar:
            for text in ff:
                with contextlib.redirect_stdout(io.StringIO()):
                    necc_pp, suff_pp, necc_mm, suff_mm = calculate_necc_and_suff(text, ilm_tokenizer=tokenizer, ilm_model=model, cl_tokenizer=None, cl_model=None, num_samples=num_samples,
                                        mask_tokn=mask_tokn, additional_tokens_to_ids=additional_tokens_to_ids, probs_table=probs_table, 
                                        return_pert_only=True)

                orig_texts.append(text)
                necc_perturbed.append(necc_pp)
                suff_perturbed.append(suff_pp)
                necc_masks.append(necc_mm)
                suff_masks.append(suff_mm)
                pbar.update(1)
        
necc_suff_perturbations = {'orig_texts': orig_texts, 
                           'necc_perturbed': necc_perturbed, 
                           'suff_perturbed': suff_perturbed,
                           'necc_masks': necc_masks,
                           'suff_masks': suff_masks}

#pickle.dump(necc_suff_perturbations, open('Data/HateCheck_necc_suff_perturbations.pickle', 'wb'))
pickle.dump(necc_suff_perturbations, open('Data/HateCheck_necc_suff_perturbations_2.pickle', 'wb'))