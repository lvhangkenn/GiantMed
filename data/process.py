import pandas as pd
import dill
import numpy as np

from collections import Counter
from chemfrag import DGLMolTree
from xml.dom.pulldom import ErrorHandler
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import BRICS
from bricsfrag import MolTree


# ==========================================medical

def med_process(med_file):
    med_pd = pd.read_csv(med_file, dtype={'NDC': 'category'})

    med_pd.drop(columns=[
        'ROW_ID', 'DRUG_TYPE', 'DRUG_NAME_POE', 'DRUG_NAME_GENERIC',
        'FORMULARY_DRUG_CD', 'PROD_STRENGTH', 'DOSE_VAL_RX',
        'DOSE_UNIT_RX', 'FORM_VAL_DISP', 'FORM_UNIT_DISP', 'GSN',
        'FORM_UNIT_DISP', 'ROUTE', 'ENDDATE'
    ], axis=1, inplace=True)
    med_pd.drop(index=med_pd[med_pd['NDC'] == '0'].index, axis=0, inplace=True)
    med_pd.fillna(method='pad', inplace=True)
    med_pd.dropna(inplace=True)
    med_pd.drop_duplicates(inplace=True)
    med_pd['ICUSTAY_ID'] = med_pd['ICUSTAY_ID'].astype('int64')
    med_pd['STARTDATE'] = pd.to_datetime(
        med_pd['STARTDATE'], format='%Y-%m-%d %H:%M:%S'
    )
    med_pd.sort_values(
        by=['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'STARTDATE', 'DRUG'], inplace=True
    )
    med_pd = med_pd.reset_index(drop=True)

    med_pd = med_pd.drop(columns=['ICUSTAY_ID'])
    med_pd = med_pd.drop_duplicates()
    med_pd = med_pd.reset_index(drop=True)
    return med_pd


def process_visit_lg2(med_pd):
    """visit >=2"""
    a = (
        med_pd[["SUBJECT_ID", "HADM_ID"]]
        .groupby(by="SUBJECT_ID")["HADM_ID"]
        .unique()
        .reset_index()
    )
    a["HADM_ID_Len"] = a["HADM_ID"].map(lambda x: len(x))
    a = a[a["HADM_ID_Len"] >= 2]  # subject id, harm id
    return a


# medication mapping
def codeMapping2atc4(med_pd):
    """ndc->rxcui->atc level code"""
    with open(ndc2RXCUI_file, "r") as f:
        ndc2RXCUI = eval(f.read())  # dict

    med_pd["RXCUI"] = med_pd["NDC"].map(ndc2RXCUI)  # NDC->RXCUI
    med_pd.dropna(inplace=True)

    RXCUI2atc4 = pd.read_csv(RXCUI2atc4_file)  # RXCUI -> atc
    RXCUI2atc4 = RXCUI2atc4.drop(columns=["YEAR", "MONTH", "NDC"])
    RXCUI2atc4.drop_duplicates(subset=["RXCUI"], inplace=True)
    med_pd.drop(index=med_pd[med_pd["RXCUI"].isin([""])].index, axis=0, inplace=True)

    med_pd["RXCUI"] = med_pd["RXCUI"].astype("int64")
    med_pd = med_pd.reset_index(drop=True)
    med_pd = med_pd.merge(RXCUI2atc4, on=["RXCUI"])
    med_pd.drop(columns=["NDC", "RXCUI"], inplace=True)

    med_pd["ATC4"] = med_pd["ATC4"].map(lambda x: x[:4])
    med_pd = med_pd.rename(columns={"ATC4": "ATC3"})
    med_pd = med_pd.drop_duplicates()
    med_pd = med_pd.reset_index(drop=True)
    return med_pd


def filter_300_most_med(med_pd):
    med_count = (
        med_pd.groupby(by=["ATC3"])
        .size()
        .reset_index()
        .rename(columns={0: "count"})
        .sort_values(by=["count"], ascending=False)
        .reset_index(drop=True)
    )
    med_pd = med_pd[med_pd["ATC3"].isin(med_count.loc[:299, "ATC3"])]
    med_pd = med_pd.drop_duplicates(subset=["HADM_ID", "ATC3"], keep="first")
    return med_pd.reset_index(drop=True)

# ==========================================diag
def diag_process(diag_file):
    """patient symptom"""
    diag_pd = pd.read_csv(diag_file)
    diag_pd.dropna(inplace=True)
    diag_pd.drop(columns=["SEQ_NUM", "ROW_ID"], inplace=True)
    diag_pd.drop_duplicates(inplace=True)
    diag_pd.sort_values(by=["SUBJECT_ID", "HADM_ID"], inplace=True)
    diag_pd = diag_pd.reset_index(drop=True)

    def filter_2000_most_diag(diag_pd):
        diag_count = (
            diag_pd.groupby(by=["ICD9_CODE"])
            .size()
            .reset_index()
            .rename(columns={0: "count"})
            .sort_values(by=["count"], ascending=False)
            .reset_index(drop=True)
        )
        diag_pd = diag_pd[diag_pd["ICD9_CODE"].isin(diag_count.loc[:1999, "ICD9_CODE"])]

        return diag_pd.reset_index(drop=True)

    diag_pd = filter_2000_most_diag(diag_pd)
    icd9_pd = pd.read_csv("raw/D_ICD_DIAGNOSES.csv")
    diag_pd = diag_pd.merge(icd9_pd[['ICD9_CODE', 'SHORT_TITLE', 'LONG_TITLE']], on='ICD9_CODE', how='inner')
    diag_pd.dropna(subset=['SHORT_TITLE', 'LONG_TITLE'], inplace=True)
    diag_pd.sort_values(by=["SUBJECT_ID", "HADM_ID", 'SHORT_TITLE', 'LONG_TITLE'], inplace=True)
    diag_pd = diag_pd.reset_index(drop=True)

    return diag_pd


# ==========================================procedure

def procedure_process(procedure_file):
    """procedure process"""
    pro_pd = pd.read_csv(procedure_file)
    pro_pd.dropna(inplace=True)
    pro_pd.drop(columns=["SEQ_NUM", "ROW_ID"], inplace=True)
    pro_pd.drop_duplicates(inplace=True)
    pro_pd.sort_values(by=["SUBJECT_ID", "HADM_ID"], inplace=True)
    pro_pd = pro_pd.reset_index(drop=True)
    icd9_pd = pd.read_csv("raw/D_ICD_PROCEDURES.csv")
    pro_pd = pro_pd.merge(icd9_pd[['ICD9_CODE', 'SHORT_TITLE', 'LONG_TITLE']], on='ICD9_CODE', how='inner')
    pro_pd.dropna(subset=['SHORT_TITLE', 'LONG_TITLE'], inplace=True)
    pro_pd.sort_values(by=["SUBJECT_ID", "HADM_ID", 'SHORT_TITLE', 'LONG_TITLE'], inplace=True)
    pro_pd = pro_pd.reset_index(drop=True)
    return pro_pd


# ==========================================combine
def combine_process(med_pd, diag_pd, pro_pd):

    med_pd_key = med_pd[["SUBJECT_ID", "HADM_ID"]].drop_duplicates()
    diag_pd_key = diag_pd[["SUBJECT_ID", "HADM_ID"]].drop_duplicates()
    pro_pd_key = pro_pd[["SUBJECT_ID", "HADM_ID"]].drop_duplicates()

    combined_key = med_pd_key.merge(diag_pd_key, on=["SUBJECT_ID", "HADM_ID"], how="inner")
    combined_key = combined_key.merge(pro_pd_key, on=["SUBJECT_ID", "HADM_ID"], how="inner")

    diag_pd = diag_pd.merge(combined_key, on=["SUBJECT_ID", "HADM_ID"], how="inner")
    med_pd = med_pd.merge(combined_key, on=["SUBJECT_ID", "HADM_ID"], how="inner")
    pro_pd = pro_pd.merge(combined_key, on=["SUBJECT_ID", "HADM_ID"], how="inner")

    diag_pd = (
        diag_pd.groupby(by=["SUBJECT_ID", "HADM_ID"])["ICD9_CODE"]
        .apply(lambda x: list(x.unique()))
        .reset_index()
    )

    med_pd = med_pd.groupby(by=["SUBJECT_ID", "HADM_ID"]).agg({
        "ATC3": "unique",
        "DRUG": "unique"
    }).reset_index()

    pro_pd = (
        pro_pd.groupby(by=["SUBJECT_ID", "HADM_ID"])["ICD9_CODE"]
        .apply(lambda x: list(x.unique()))
        .reset_index()
        .rename(columns={"ICD9_CODE": "PRO_CODE"})
    )

    med_pd["ATC3"] = med_pd["ATC3"].map(lambda x: list(x))
    med_pd["DRUG"] = med_pd["DRUG"].map(lambda x: list(x) if isinstance(x, list) else [x])  # Ensure DRUG is a list
    pro_pd["PRO_CODE"] = pro_pd["PRO_CODE"].map(lambda x: list(x))

    data = diag_pd.merge(med_pd, on=["SUBJECT_ID", "HADM_ID"], how="inner")
    data = data.merge(pro_pd, on=["SUBJECT_ID", "HADM_ID"], how="inner")

    data["ATC3_num"] = data["ATC3"].map(lambda x: len(x))
    data["DRUG_num"] = data["DRUG"].map(lambda x: len(x))  # Count unique drugs

    data.dropna(inplace=True)

    return data


def statistics(data):

    diag = data["ICD9_CODE"].values
    med = data["ATC3"].values
    pro = data["PRO_CODE"].values

    unique_diag = set([j for i in diag for j in list(i)])
    unique_med = set([j for i in med for j in list(i)])
    unique_pro = set([j for i in pro for j in list(i)])

    print("#diagnosis ", len(unique_diag))
    print("#med ", len(unique_med))
    print("#procedure", len(unique_pro))

    (
        avg_diag,
        avg_med,
        avg_pro,
        max_diag,
        max_med,
        max_pro,
        cnt,
        max_visit,
        avg_visit,
    ) = [0 for i in range(9)]

    for subject_id in data["SUBJECT_ID"].unique():
        item_data = data[data["SUBJECT_ID"] == subject_id]
        x, y, z = [], [], []
        visit_cnt = 0
        for index, row in item_data.iterrows():
            visit_cnt += 1
            cnt += 1
            x.extend(list(row["ICD9_CODE"]))
            y.extend(list(row["ATC3"]))
            z.extend(list(row["PRO_CODE"]))
        x, y, z = set(x), set(y), set(z)
        avg_diag += len(x)
        avg_med += len(y)
        avg_pro += len(z)
        avg_visit += visit_cnt
        if len(x) > max_diag:
            max_diag = len(x)
        if len(y) > max_med:
            max_med = len(y)
        if len(z) > max_pro:
            max_pro = len(z)
        if visit_cnt > max_visit:
            max_visit = visit_cnt
    print("cnt",cnt)
    print(avg_med)
    print("#avg of diagnoses ", avg_diag / cnt)
    print("#avg of medicines ", avg_med / cnt)
    print("#avg of procedures ", avg_pro / cnt)
    print("#avg of vists ", avg_visit / len(data["SUBJECT_ID"].unique()))

    print("#max of diagnoses ", max_diag)
    print("#max of medicines ", max_med)
    print("#max of procedures ", max_pro)
    print("#max of visit ", max_visit)


# ==========================================voc set; index-reorder + save final record

class Voc(object):
    """re_encode"""

    def __init__(self):
        self.idx2word = {}
        self.word2idx = {}

    def add_sentence(self, sentence):
        for word in sentence:
            if word not in self.word2idx:
                self.idx2word[len(self.word2idx)] = word
                self.word2idx[word] = len(self.word2idx)


def create_str_token_mapping(df):
    """create voc set"""
    diag_voc = Voc()
    med_voc = Voc()
    pro_voc = Voc()

    for index, row in df.iterrows():
        diag_voc.add_sentence(row["ICD9_CODE"])
        med_voc.add_sentence(row["ATC3"])
        pro_voc.add_sentence(row["PRO_CODE"])

    dill.dump(
        obj={"diag_voc": diag_voc, "med_voc": med_voc, "pro_voc": pro_voc},
        file=open(vocabulary_file, "wb"),
    )
    return diag_voc, med_voc, pro_voc

def create_patient_record(df, diag_voc, med_voc, pro_voc):
    """# create final records"""

    records = []  # (patient, code_kind:3, codes)  code_kind:diag, proc, med
    for subject_id in df["SUBJECT_ID"].unique():
        item_df = df[df["SUBJECT_ID"] == subject_id]
        patient = []
        for index, row in item_df.iterrows():
            admission = []
            admission.append([diag_voc.word2idx[i] for i in row["ICD9_CODE"]])
            admission.append([pro_voc.word2idx[i] for i in row["PRO_CODE"]])
            admission.append([med_voc.word2idx[i] for i in row["ATC3"]])
            admission.append([subject_id])
            admission.append([row['HADM_ID']])
            patient.append(admission)
        records.append(patient)
    dill.dump(obj=records, file=open(ehr_sequence_file, "wb"))
    ratio = 2 / 3
    split_point = int(len(records) * ratio)
    eval_len = int(len(records[split_point:]) / 2)
    data_test = records[split_point: split_point + eval_len]  # valid
    data_eval = records[split_point + eval_len:]  # test

    total_medicines = 0
    total_visits = 0

    for patient in data_test:
        for visit in patient:
            medicines = visit[2]
            total_medicines += len(medicines)
            total_visits += 1

    # 计算平均药物数量
    avg_medicines = total_medicines / total_visits if total_visits > 0 else 0

    print(f"Average number of medicines per visit: {avg_medicines}")
    print("statistics(data_test)")
    return records

def get_ddi_matrix(records, med_voc, ddi_file):
    """EHR data; DDI side effect"""
    TOPK = 40  # topk drug-drug interaction
    cid2atc_dic = defaultdict(set)
    med_voc_size = len(med_voc.idx2word)
    med_unique_word = [med_voc.idx2word[i] for i in range(med_voc_size)]  # ATC3
    atc3_atc4_dic = defaultdict(set)
    for item in med_unique_word:
        atc3_atc4_dic[item[:4]].add(item)
    with open(cid2atc6_file, "r") as f:
        for line in f:
            line_ls = line[:-1].split(",")
            cid = line_ls[0]
            atcs = line_ls[1:]
            for atc in atcs:
                if len(atc3_atc4_dic[atc[:4]]) != 0:
                    cid2atc_dic[cid].add(atc[:4])

    # ddi load
    ddi_df = pd.read_csv(ddi_file)
    ddi_most_pd = (
        ddi_df.groupby(by=["Polypharmacy Side Effect", "Side Effect Name"])
        .size()
        .reset_index()
        .rename(columns={0: "count"})
        .sort_values(by=["count"], ascending=False)
        .reset_index(drop=True)
    )
    ddi_most_pd = ddi_most_pd.iloc[-TOPK:, :]
    # ddi_most_pd = pd.DataFrame(columns=['Side Effect Name'], data=['as','asd','as'])
    fliter_ddi_df = ddi_df.merge(
        ddi_most_pd[["Side Effect Name"]], how="inner", on=["Side Effect Name"]
    )
    ddi_df = (
        fliter_ddi_df[["STITCH 1", "STITCH 2"]].drop_duplicates().reset_index(drop=True)
    )

    ehr_adj = np.zeros((med_voc_size, med_voc_size))
    for patient in records:
        for adm in patient:
            med_set = adm[2]
            for i, med_i in enumerate(med_set):
                for j, med_j in enumerate(med_set):
                    if j <= i:
                        continue
                    ehr_adj[med_i, med_j] = 1
                    ehr_adj[med_j, med_i] = 1
    dill.dump(ehr_adj, open(ehr_adjacency_file, "wb"))

    # ddi adj
    ddi_adj = np.zeros((med_voc_size, med_voc_size))
    for index, row in ddi_df.iterrows():
        # ddi
        cid1 = row["STITCH 1"]
        cid2 = row["STITCH 2"]

        # cid -> atc_level3
        for atc_i in cid2atc_dic[cid1]:
            for atc_j in cid2atc_dic[cid2]:

                # atc_level3 -> atc_level4
                for i in atc3_atc4_dic[atc_i]:
                    for j in atc3_atc4_dic[atc_j]:
                        if med_voc.word2idx[i] != med_voc.word2idx[j]:
                            ddi_adj[med_voc.word2idx[i], med_voc.word2idx[j]] = 1
                            ddi_adj[med_voc.word2idx[j], med_voc.word2idx[i]] = 1
    dill.dump(ddi_adj, open(ddi_adjacency_file, "wb"))

    return ddi_adj


def get_ddi_mask(atc42SMLES, med_voc):
    """drug-substructure"""
    fraction = []
    for k, v in med_voc.idx2word.items():  # id:atc
        tempF = set()
        for SMILES in atc42SMLES[v]:
            try:
                m = BRICS.BRICSDecompose(Chem.MolFromSmiles(SMILES))
                for frac in m:
                    tempF.add(frac)
            except:
                pass
        fraction.append(tempF)  # [(),()]substructure
    fracSet = []
    for i in fraction:
        fracSet += i
    fracSet = list(set(fracSet))  # set of all segments

    ddi_matrix = np.zeros((len(med_voc.idx2word), len(fracSet)))
    for i, fracList in enumerate(fraction):
        for frac in fracList:
            ddi_matrix[i, fracSet.index(frac)] = 1
    return ddi_matrix


# ======================================================new

def get_drug_smile_relation(atc42SMLES, med_voc):
    temp = set()
    for k, v in med_voc.idx2word.items():  # id:atc
        for SMILES in atc42SMLES[v]:
            temp.add(SMILES)
    smiles = list(temp)
    drug_smile_matrix = np.zeros((len(med_voc.idx2word), len(smiles)))
    for k, v in med_voc.idx2word.items():
        for smile in atc42SMLES[v]:
            drug_smile_matrix[k, smiles.index(smile)] = 1
    return smiles, drug_smile_matrix


def get_smile_subs_relation(smiles):
    """进行结构分解，非brics"""
    subs = set()
    smiles_recency = []
    smiles_degree = []
    for smile in smiles:
        smile_graph = DGLMolTree(smiles=smile)
        data = smile_graph.nodes_dict
        if data == {}:
            print(smile)
        subs_dict = {key: node['smiles'] for key, node in data.items()}  # {0: 'CC'}
        degree = list(smile_graph.in_degrees().numpy().astype(int))  # [0,1,2,3]

        recency = dict(Counter(node['smiles'] for node in data.values()))
        max_degrees = {value: max(degree[key] for key, val in subs_dict.items() if val == value) for value in
                       set(subs_dict.values())}  # {substructure:degree}
        smiles_recency.append(recency)
        smiles_degree.append(max_degrees)
        for key in recency.keys():
            subs.add(key)

    # feature & matrix 提取
    subs = ['uknown'] + list(subs)
    print('Subs, ', subs)
    smile_subs_matrix = np.zeros((len(smiles), len(subs)))
    smile_sub_degree = np.zeros((len(smiles), len(subs)))
    smile_sub_recency = np.zeros((len(smiles), len(subs)))
    print("Recency, ", smiles_recency)
    print("Degree, ", smiles_degree)
    # min_degree = []
    for i in range(len(smiles)):
        recency = smiles_recency[i]
        degree = smiles_degree[i]
        # min_degree.append(min(degree.values()))
        for j in recency.keys():
            smile_subs_matrix[i, subs.index(j)] = 1
            smile_sub_degree[i, subs.index(j)] = degree[j] + 1
            smile_sub_recency[i, subs.index(j)] = recency[j]
    # print(min_degree)
    # print(min(min_degree))

    return subs, smile_subs_matrix, smile_sub_degree, smile_sub_recency


def get_smile_subs_relation_brics(smiles):

    subs = set()
    smiles_recency = []
    smiles_degree = []
    for smile in smiles:
        smile_graph = MolTree(smiles=smile)
        # smile_graph.recover()
        # smile_graph.assemble()

        subs_dict = {c.nid - 1: c.smiles for c in smile_graph.nodes}  # {0: 'CC'}
        if subs_dict == {}:
            print('不能切分的子集', smile)
            subs_dict = {0: 'unknown'}
            recency = {'unknown': 1}
            smiles_recency.append(recency)
            smiles_degree.append({'unknown': 0})
        else:
            degree = [len(c.neighbors) for c in smile_graph.nodes]
            recency = {}
            for c in smile_graph.nodes:
                if c.smiles not in recency:
                    recency[c.smiles] = 1
                else:
                    recency[c.smiles] += 1

            max_degrees = {value: max(degree[key] for key, val in subs_dict.items() if val == value) for value in
                           set(subs_dict.values())}  # {substructure:degree}
            smiles_recency.append(recency)
            smiles_degree.append(max_degrees)
        for key in recency.keys():
            subs.add(key)

    subs = ['uknown'] + list(subs)

    print('Subs, ', subs)
    smile_subs_matrix = np.zeros((len(smiles), len(subs)))
    smile_sub_degree = np.zeros((len(smiles), len(subs)))
    smile_sub_recency = np.zeros((len(smiles), len(subs)))
    print("Recency, ", smiles_recency)
    print("Degree, ", smiles_degree)
    # min_degree = []
    for i in range(len(smiles)):
        recency = smiles_recency[i]
        degree = smiles_degree[i]
        # min_degree.append(min(degree.values()))
        for j in recency.keys():
            smile_subs_matrix[i, subs.index(j)] = 1
            smile_sub_degree[i, subs.index(j)] = degree[j] + 1
            smile_sub_recency[i, subs.index(j)] = recency[j]


    return subs, smile_subs_matrix, smile_sub_degree, smile_sub_recency


def filter_failure(my_dict):
    # lis = ['H[N]1[C@H]2CCCC[C@@H]2[N](H)(H)[Pt]11OC(=O)C(=O)O1','[Cl-].[Cl-].[Ca++]','[K+].[I-]',
    #                       '[H][N]1([H])[C@@H]2CCCC[C@H]2[N]([H])([H])[Pt]11OC(=O)C(=O)O1',

    lis = ['[H][N]([H])([H])[Pt]1(OC(=O)C2(CCC2)C(=O)O1)[N]([H])([H])[H]',
           # '[OH-].[OH-].[Mg++]','[F-].[Na+]','[Na+].[Cl-]'
           ]
    for key, value in my_dict.items():
        my_dict[key] = [item for item in value if item not in lis]
    keys_to_delete = []
    for key, value in my_dict.items():
        if not value:
            keys_to_delete.append(key)
    for key in keys_to_delete:
        my_dict.pop(key)
    return my_dict


if __name__ == '__main__':
    root = "raw/MIMIC-III"
    root_to = "ready/MIMIC-III"
    print("=================Start!=================")
    # input file
    med_file = root + "PRESCRIPTIONS.csv"
    diag_file = root + "DIAGNOSES_ICD.csv"
    procedure_file = root + "PROCEDURES_ICD.csv"

    RXCUI2atc4_file = root + "ndc2atc_level4.csv"

    cid2atc6_file = root + "drug-atc.csv"
    ndc2RXCUI_file = root + "ndc2RXCUI.txt"
    ddi_file = root + "drug-DDI.csv"
    drugbankinfo = root + "drugbank_drugs_info.csv"
    med_structure_file = root + "idx2SMILES.pkl"

    # output file
    ddi_adjacency_file = root_to + "ddi_A_final.pkl"
    ehr_adjacency_file = root_to + "ehr_adj_final.pkl"

    ehr_sequence_file = root_to + "records_final.pkl"
    vocabulary_file = root_to + "voc_final.pkl"
    ddi_mask_H_file = root_to + "ddi_mask_H.pkl"
    atc3toSMILES_file = root_to + "atc3toSMILES.pkl"


    # for drug process
    med_pd = med_process(med_file)
    med_pd_lg2 = process_visit_lg2(med_pd).reset_index(drop=True)  # visit > 2
    med_pd = med_pd.merge(
        med_pd_lg2[["SUBJECT_ID"]], on="SUBJECT_ID", how="inner"
    ).reset_index(drop=True)
    med_pd = codeMapping2atc4(med_pd)

    atc3toSMILES = dill.load(open(med_structure_file, 'rb'))
    atc3toSMILES = filter_failure(atc3toSMILES)

    med_pd = med_pd[med_pd.ATC3.isin(atc3toSMILES.keys())]
    dill.dump(atc3toSMILES, open(atc3toSMILES_file, "wb"))
    med_pd = filter_300_most_med(med_pd)

    # for diagnosis
    diag_pd = diag_process(diag_file)

    # for procedure
    pro_pd = procedure_process(procedure_file)

    # combine
    data = combine_process(med_pd, diag_pd, pro_pd)

    # create vocab
    diag_voc, med_voc, pro_voc = create_str_token_mapping(data)
    print("obtain voc")

    # create ehr sequence data
    records = create_patient_record(data, diag_voc, med_voc, pro_voc)
    print("obtain ehr sequence data")

    # create ddi adj matrix ddi
    ddi_adj = get_ddi_matrix(records, med_voc, ddi_file)
    print("obtain ddi adj matrix")

    ddi_mask_H = get_ddi_mask(atc3toSMILES, med_voc)
    dill.dump(ddi_mask_H, open(ddi_mask_H_file, "wb"))

    # get drug-smile matrix
    drug_smile_file = root_to + "drug_smile.pkl"
    smiles, drug_smile_matrix = get_drug_smile_relation(atc3toSMILES, med_voc)  # smiles: list
    smile_voc = dict(zip(smiles, list(range(len(smiles)))))
    print("obtain drug-smile matrix")
    dill.dump(drug_smile_matrix, open(drug_smile_file, "wb"))

    # get smile substructure voc
    smile_subs_file = root_to + "smile_sub_b.pkl"
    smile_sub_degree_file = root_to + "smile_sub_degree_b.pkl"
    smiles_subs_receny_file = root_to + "smile_sub_recency_b.pkl"
    sub_voc, smile_subs_matrix, smile_sub_degree, smile_sub_recency = get_smile_subs_relation_brics(smiles)
    print("obtain smile-sub matrix")
    dill.dump(smile_subs_matrix, open(smile_subs_file, "wb"))
    dill.dump(smile_sub_degree, open(smile_sub_degree_file, "wb"))
    dill.dump(smile_sub_recency, open(smiles_subs_receny_file, "wb"))

    print("XXXXXXXXXXX", smile_subs_matrix.shape)

    voc_file = root_to + "smile_sub_voc_b.pkl"
    dill.dump(
        obj={"smile_voc": smile_voc, "sub_voc": sub_voc},
        file=open(voc_file, "wb"),
    )
    print("=================Done!===================")
