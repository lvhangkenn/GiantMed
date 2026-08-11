#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/5/29 13:45
# @Author  : Anonymous
# @Site    :
# @File    : main_MedAlign.py
# @Software: PyCharm

# #Desc: first run preprocess

import os
import torch
from config import config
from utils import seed_everything, dill_load
from trainer_MedAlign import DrugTrainer


# from draw.test import buildMPNN
def run_single_model(config):
    print("Hello MedAlign!")
    print(config)

    if config['USE_CUDA']==True:
        #os.environ["CUDA_VISIBLE_DEVICES"] = config['GPU']
        device = torch.device("cuda:0"
                              "".format(config['GPU']))
        print(device)
        print(torch.tensor(0).to(device))
        os.environ['CUDA_VISIBLE_DEVICES'] = str(device)

        print(os.environ['CUDA_VISIBLE_DEVICES'])
    else:
        device = torch.device('cpu')
    torch.cuda.set_device(device)
    print(device)
    # data initial
    #a="ready/"
    a="ready/"
    data_path = config['ROOT'] + a + "records_final.pkl"
    voc_path = config['ROOT'] + a + "voc_final.pkl" # three dict tuple;
    ddi_adj_path = config['ROOT'] + a + "ddi_A_final.pkl" # side effect matrix
    ddi_mask_path = config['ROOT'] + a + "ddi_mask_H.pkl" # substructrue-drug
    molecule_path = config['ROOT'] + a + "atc3toSMILES.pkl" # atc(drug) to smiles lis

    drug_smile_path = config['ROOT'] + a + "drug_smile.pkl"
    smile_subs_path = (config['ROOT'] +
                       a + "smile_sub_b.pkl")
    smile_sub_voc_path = config['ROOT'] + a + "smile_sub_voc_b.pkl" # two dict tuple;
    smile_sub_degree_path = config['ROOT'] + a + "smile_sub_degree_b.pkl"
    smile_sub_recency_path = config['ROOT'] + a + "smile_sub_recency_b.pkl"
    drug_text_embs_path = config['ROOT'] + a + "drug_text_embs.pkl"
    ddi_adj, ddi_mask_H, data, molecule, voc = dill_load(ddi_adj_path), dill_load(ddi_mask_path), dill_load(data_path), dill_load(molecule_path), dill_load(voc_path)
    drug_smile_matrix, smile_subs_matrix, smile_sub_voc, smile_sub_degree, smile_sub_recency,drug_text_embs= dill_load(drug_smile_path), dill_load(smile_subs_path), dill_load(smile_sub_voc_path), dill_load(smile_sub_degree_path), dill_load(smile_sub_recency_path),dill_load(drug_text_embs_path)
    # model initial
    trainer = DrugTrainer(config, device, (ddi_adj, ddi_mask_H, data, molecule, voc), (drug_smile_matrix, smile_subs_matrix, smile_sub_voc, smile_sub_degree, smile_sub_recency),
                          drug_text_embs)

    trainer.test()

    print("Everything is OK!")


if __name__ == '__main__':
    config = config
    seed_everything(config['SEED'])
    run_single_model(config)
