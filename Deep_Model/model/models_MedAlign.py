import pickle
from tkinter import Variable

import torch
torch.autograd.set_detect_anomaly(True)
import ot
torch.cuda.empty_cache()
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from torch.nn.utils.rnn import pad_sequence
from sklearn.decomposition import PCA

class MLP_Decoder(nn.Module):
    def __init__(self, hdim, nclass):
        super(MLP_Decoder, self).__init__()
        self.final_layer = nn.Linear(hdim, nclass)
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()

    def forward(self, h):
        output = self.final_layer(h)
        return output


class Drug_MLP(nn.Module):
    def __init__(self, drug_dim, n_class):
        super(Drug_MLP, self).__init__()
        self.decoder = MLP_Decoder(drug_dim, n_class)

    def forward(self, drug_dim):
        pred = self.decoder(drug_dim)
        return pred



class TransformerEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(TransformerEncoder, self).__init__()
        self.encoder = nn.TransformerEncoderLayer(d_model=input_size, nhead=2, dim_feedforward=hidden_size)
        self.transformer = nn.TransformerEncoder(self.encoder, num_layers=num_layers)

    def forward(self, x, src_mask):
        """
        :param x: [Smile_num, seq_len, dim]: seq_len=max(sub_num)
        :param src_mask: [Smile_num, seq_len]: seq_len=max(sub_num): bool, mask为true
        :return: [Smile_num, embedding]
        """
        x = x.transpose(0, 1)  # seq_len, batch, dim
        x = self.transformer(x, src_key_padding_mask=src_mask)

        x = x.transpose(0, 1)  # b,seq, dim
        mask = ~src_mask
        avg = torch.sum(mask, dim=1).unsqueeze(dim=1)  # smile_num, 1
        x = torch.bmm(mask.unsqueeze(1).float(), x).squeeze()  # Smile_num, dim;
        x = x / avg
        return x


class SubGraphNetwork(nn.Module):
    def __init__(self, sub_embed, emb_dim, smile_num):
        super(SubGraphNetwork, self).__init__()
        self.sub_emb = sub_embed  # sub_num = [unknown + voc]

        self.transformer = TransformerEncoder(emb_dim, emb_dim, 2)
        self.smile_emb_id = nn.Embedding(smile_num, emb_dim)
        self.recency_emb = nn.Embedding(30, emb_dim)
        self.degree_emb = nn.Embedding(30, emb_dim)
        self.dropout = nn.Dropout(p=0.7, inplace=False)

        self.init_weights()

        self.ff = nn.Linear(emb_dim, emb_dim, bias=False)
        self.ff2 = nn.Linear(emb_dim, emb_dim, bias=False)

    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        nn.init.xavier_uniform_(self.smile_emb_id.weight)
        nn.init.xavier_uniform_(self.recency_emb.weight)
        nn.init.xavier_uniform_(self.degree_emb.weight)

    def get_indices(self, adjacency_matrix):
        nonzero_column_indices = torch.nonzero(adjacency_matrix, as_tuple=False)[:, 1]
        lens = list(torch.sum(adjacency_matrix, dim=1).long().cpu().numpy())
        return nonzero_column_indices, lens

    def get_nonzero_values(self, matrix):
        nonzero_values = torch.masked_select(matrix, matrix != 0)
        return nonzero_values

    def get_embedding(self, smile_sub_matrix, smile_sub_recency, smile_sub_degree):
        # smile-sub indice
        sub_ids, lens = self.get_indices(smile_sub_matrix)  # [],
        # print(len(sub_ids))
        smile_sub_seq = [v for v in torch.split(sub_ids, lens)]  # [[],[]]
        padded_sequences = pad_sequence(smile_sub_seq, batch_first=True, padding_value=0).long()  # [smile, max_subs+1]
        sub_embs = self.sub_emb(padded_sequences)  # smile_num, max_sub_num+1, dim
        mask = (padded_sequences == 0).bool()

        # structure embedding
        smile_sub_recency = self.get_nonzero_values(smile_sub_recency)
        smile_sub_recency = [torch.sort(v)[1] for v in torch.split(smile_sub_recency, lens)]
        padded_sequences = pad_sequence(smile_sub_recency, batch_first=True, padding_value=0).long()
        recency_embs = self.recency_emb(padded_sequences)  # smile_num, sub_num+1, dim

        smile_sub_degree = self.get_nonzero_values(smile_sub_degree)
        smile_sub_degree = [torch.sort(v)[1] for v in torch.split(smile_sub_degree, lens)]
        padded_sequences = pad_sequence(smile_sub_degree, batch_first=True, padding_value=0).long()
        degree_embs = self.degree_emb(padded_sequences)  # smile_num, sub_num+1, dim

        # intergrate
        emb = self.dropout(self.ff(recency_embs + degree_embs)) + sub_embs

        return emb, mask

    def graph_transformer(self, input, src_mask):
        """
        input: [sub,sub,sub], [pos, pos, pos];
        :return:
        """
        smile_rep = self.transformer(input, src_mask)
        return smile_rep

    def sum(self, smile_embs, split_point):
        """
        split_point: list[3,4,2]
        """
        sum_vectors = [torch.sum(v, 0) for v in torch.split(smile_embs, split_point)]
        return torch.stack(sum_vectors)  # molecular , B

    def forward(self, inputs, query=None):
        """
        sub_dicts:,
        smile_sub_matrix:,
        adjacencies:,
        molecular_sizes:
        """
        smile_sub_matrix, recency_matrix, degree_matrix, drug_smile_matrix = inputs
        emb, mask = self.get_embedding(smile_sub_matrix, recency_matrix, degree_matrix)
        smile_rep = self.graph_transformer(emb, mask)
        smile_rep = smile_rep + self.smile_emb_id.weight
        drug_attn = F.softmax(torch.matmul(query, smile_rep.t()), dim=-1)
        drug_rep = drug_attn.matmul(smile_rep)
        drug_rep = F.normalize(drug_rep, p=2, dim=1)

        return drug_rep


class DrugRec(nn.Module):
    def __init__(self, vocab_size, emb_dim, drug_smile, smile_sub, ddi_matrix, structure_matrix, device,
                 drug_text_embs):
        super(DrugRec, self).__init__()
        print(vocab_size[-2], vocab_size[-1])
        self.vocab_size=vocab_size
        self.emb_dim = emb_dim
        self.device = device
        self.diagnose_emb = nn.Embedding(vocab_size[0], emb_dim)
        self.procedure_emb = nn.Embedding(vocab_size[1], emb_dim)

        self.dropout = nn.Dropout(p=0.7, inplace=False)
        self.diagnose_encoder = nn.GRU(emb_dim, emb_dim, batch_first=True)
        self.procedure_encoder = nn.GRU(emb_dim, emb_dim, batch_first=True)
        self.his_encoder = nn.GRU(emb_dim, emb_dim, batch_first=True)

        self.drug_text_embs = drug_text_embs
        self.init_weights()

        self.flow_structure2id = None
        self.flow_text2id = None
        self.drug_structure_emb = None

        self.drug_mlp = Drug_MLP(64, 131).to(self.device)

        self.drug_smile = drug_smile  # matrix
        self.smile_sub = smile_sub  # matrix
        self.sub_embed = nn.Embedding(vocab_size[-1], emb_dim)

        self.query = nn.Sequential(nn.ReLU(), nn.Linear(3 * emb_dim, emb_dim))
        self.query_plm = nn.Sequential(nn.ReLU(), nn.Linear(12 * emb_dim, 1 * emb_dim))

        self.W1 = nn.Linear(emb_dim, emb_dim, bias=False)  # Linear transformation for H1
        self.W2 = nn.Linear(emb_dim, emb_dim, bias=False)  # Linear transformation for H2

        # # graph embedding
        self.graph_network = SubGraphNetwork(self.sub_embed, emb_dim, self.drug_smile.shape[1]).to(self.device)
        self.stru = structure_matrix

        # ddi
        self.ddi_matrix = ddi_matrix
        self.drug_emb_id = nn.Embedding(self.drug_smile.shape[0], self.emb_dim)

    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1

        # nn.init.xavier_uniform_(self.query.weight)
        # nn.init.xavier_uniform_(self.diagnose_emb.weight)
        # nn.init.xavier_uniform_(self.procedure_emb.weight)
        # nn.init.xavier_uniform_(self.sub_embed.weight)
        # nn.init.xavier_uniform_(self.drug_emb_id.weight)
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)

        self.apply(_init_weights)

    def forward(self, input):
        """
        :param input: patient health [[[],[],[]], [[],[],[]]]
        :param drug_his: patient drug history

        :return: prob, ddi rate
        """
        self.drug_emb_id = self.drug_emb_id
        self.drug_text_embs = self.drug_text_embs.to(self.device)

        self.drug_text_emb = self.query_plm(self.drug_text_embs).to(self.device)
        self.drug_structure_emb = self.graph_network([self.smile_sub, self.stru[0], self.stru[1], self.drug_smile],
                                                     self.drug_emb_id.weight)

        ot_structure2id = torch.transpose(self.flow_structure2id, 0, 1) @ self.drug_structure_emb
        ot_text2id = torch.transpose(self.flow_text2id, 0, 1) @ self.drug_text_emb

        # H = torch.stack([ot_structure2id, ot_text2id, self.drug_emb_id.weight], dim=0)
        # H1 = self.W1(H)
        # combined_emb = (ot_structure2id + ot_text2id + self.drug_emb_id.weight) / 3
        # H2 = self.W2(combined_emb.unsqueeze(0))
        # attn_scores = torch.matmul(H1, H2.transpose(-1, -2)) / torch.sqrt(torch.tensor(H1.size(-1), dtype=torch.float))
        # w_c = F.softmax(attn_scores, dim=0)
        # h_structure = torch.sum(w_c[0].unsqueeze(-1) * ot_structure2id, dim=0)  # Shape: [64]
        # h_text = torch.sum(w_c[1].unsqueeze(-1) * ot_text2id, dim=0)
        # h_drug = torch.sum(w_c[2].unsqueeze(-1) * self.drug_emb_id.weight, dim=0)

        mean = (ot_structure2id + ot_text2id + self.drug_emb_id.weight) / 3
        self.drug_emb = mean

        def sum_embedding(embedding):
            return embedding.sum(dim=1).unsqueeze(dim=0)  # (1,1,dim)

        # patient health representation
        diag_seq = []
        proc_seq = []
        his_med_seq = []

        history_meds = []

        for adm in input:
            if history_meds:
                his_med_emb = sum_embedding(
                    self.dropout(
                        self.drug_emb[torch.LongTensor(history_meds).unsqueeze(0).to(self.device)]
                    )
                )
            else:
                his_med_emb = torch.zeros(1, 1, 64).to(self.device)

            his_med_seq.append(his_med_emb)

            diag = sum_embedding(
                self.dropout(
                    self.diagnose_emb(torch.LongTensor(adm[0]).unsqueeze(0).to(self.device))
                )
            )  # (1,1,dim)
            proc = sum_embedding(
                self.dropout(
                    self.procedure_emb(torch.LongTensor(adm[1]).unsqueeze(0).to(self.device))
                )
            )

            diag_seq.append(diag)
            proc_seq.append(proc)
            history_meds.extend(adm[-1])

        diag_seq = torch.cat(diag_seq, dim=1)  # (1,seq,dim)
        proc_seq = torch.cat(proc_seq, dim=1)  # (1,seq,dim)
        his_med_seq = torch.cat(his_med_seq, dim=1) if his_med_seq else torch.zeros(1, 1, 64).to(self.device)

        o1, h1 = self.diagnose_encoder(diag_seq)  # o: all，h: last
        o2, h2 = self.procedure_encoder(proc_seq)
        o3, h3 = self.his_encoder(his_med_seq)

        visit = torch.cat([o1, o2, o3], dim=-1).squeeze(dim=0)

        his_query = self.query(visit)
        query = his_query[-1:, :]

        result = self.drug_mlp(query)

        neg_pred_prob = torch.sigmoid(result)
        neg_pred_prob = neg_pred_prob.t() * neg_pred_prob  # (voc_size, voc_size)
        ddi_rate = 0.0005 * neg_pred_prob.mul(self.ddi_matrix).sum()

        return result, ddi_rate, query

    def ot(self):
        drug_structure_emb = self.graph_network([self.smile_sub, self.stru[0], self.stru[1], self.drug_smile],
                                                self.drug_emb_id.weight)
        drug_emb_id = self.drug_emb_id
        drug_text_embs = self.drug_text_embs.to(self.device)
        drug_text_emb = self.query_plm(drug_text_embs).to(self.device)

        with torch.no_grad():
            weight_i_structure = Variable(1. / drug_structure_emb.shape[0] *
                                          torch.FloatTensor(drug_structure_emb.shape[0]).fill_(1),
                                          requires_grad=False).cuda().to(self.device)

            weight_i_text = Variable(1. / drug_text_emb.shape[0] *
                                     torch.FloatTensor(drug_text_emb.shape[0]).fill_(1), requires_grad=False).cuda().to(
                self.device)
            drug_emb_weights = drug_emb_id.weight
            weight_i_id = Variable(1. / drug_emb_weights.shape[0] *
                                   torch.FloatTensor(drug_emb_weights.shape[0]).fill_(1), requires_grad=False).to(
                self.device)

            norm_structure_id = torch.sqrt(torch.norm(drug_structure_emb, 'fro')).item() * torch.sqrt(
                torch.norm(drug_emb_weights, 'fro')).item()
            norm_text_id = torch.sqrt(torch.norm(drug_text_emb, 'fro')).item() * torch.sqrt(
                torch.norm(drug_emb_weights, 'fro')).item()

            cost_matrix_structure2id = 1 - drug_structure_emb @ torch.transpose(drug_emb_weights, 0,
                                                                                1) / norm_structure_id
            cost_matrix_text2id = 1 - drug_text_emb @ torch.transpose(drug_emb_weights, 0, 1) / norm_text_id

            flow_structure2id = ot.sinkhorn(weight_i_structure, weight_i_id, cost_matrix_structure2id, 50,
                                            numItermax=1000)
            flow_text2id = ot.sinkhorn(weight_i_text, weight_i_id, cost_matrix_text2id, 50, numItermax=1000)

            self.flow_structure2id = flow_structure2id
            self.flow_text2id = flow_text2id
            del weight_i_structure, weight_i_text, cost_matrix_structure2id, cost_matrix_text2id
            torch.cuda.empty_cache()

