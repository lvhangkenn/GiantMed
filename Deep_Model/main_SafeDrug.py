import dill
import numpy as np
import argparse
from collections import defaultdict
from torch.optim import Adam
import os
import torch
import time
from Deep_Model.model.models_SafeDrug import SafeDrugModel
from util import llprint, multi_label_metric, ddi_rate_score, buildMPNN
import torch.nn.functional as F
import json  # 新增: 用于处理JSON文件
import heapq # 新增: 用于获取top-k预测

torch.manual_seed(1203)
np.random.seed(2048)

# setting
model_name = "SafeDrug"
resume_path = "ckpt_SafeDrug.model"

if not os.path.exists(os.path.join("saved", model_name)):
    os.makedirs(os.path.join("saved", model_name))

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument("--Test", action="store_true", default=True, help="test mode")
parser.add_argument("--model_name", type=str, default=model_name, help="model name")
parser.add_argument("--resume_path", type=str, default=resume_path, help="resume path")
parser.add_argument("--lr", type=float, default=5e-4, help="learning rate")
parser.add_argument("--target_ddi", type=float, default=0.06, help="target ddi")
parser.add_argument("--kp", type=float, default=0.05, help="coefficient of P signal")
parser.add_argument("--dim", type=int, default=64, help="dimension")
parser.add_argument("--cuda", type=int, default=0, help="which cuda")
# 新增: 用于生成JSON预测报告的参数
parser.add_argument("--generate_json", action="store_true", default=True, help="Generate detailed prediction JSON file in test mode")
parser.add_argument("--num_patients", type=int, default=1058, help="Number of patients to generate predictions for")
parser.add_argument("--json_path", type=str, default="../LLM-data/SafeDrug_predictions.json", help="Output path for the detailed predictions JSON file")


args = parser.parse_args()

# evaluate
def eval(model, data_eval, voc_size, epoch):
    model.eval()

    smm_record = []
    ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]
    med_cnt, visit_cnt = 0, 0

    for step, input in enumerate(data_eval):
        y_gt, y_pred, y_pred_prob, y_pred_label = [], [], [], []
        for adm_idx, adm in enumerate(input):
            target_output, _ = model(input[: adm_idx + 1])

            y_gt_tmp = np.zeros(voc_size[2])
            y_gt_tmp[adm[2]] = 1
            y_gt.append(y_gt_tmp)

            # prediction prod
            target_output = F.sigmoid(target_output).detach().cpu().numpy()[0]
            y_pred_prob.append(target_output)

            # prediction med set
            y_pred_tmp = target_output.copy()
            y_pred_tmp[y_pred_tmp >= 0.5] = 1
            y_pred_tmp[y_pred_tmp < 0.5] = 0
            y_pred.append(y_pred_tmp)

            # prediction label
            y_pred_label_tmp = np.where(y_pred_tmp == 1)[0]
            y_pred_label.append(sorted(y_pred_label_tmp))
            visit_cnt += 1
            med_cnt += len(y_pred_label_tmp)

        smm_record.append(y_pred_label)
        adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = multi_label_metric(
            np.array(y_gt), np.array(y_pred), np.array(y_pred_prob)
        )

        ja.append(adm_ja)
        prauc.append(adm_prauc)
        avg_p.append(adm_avg_p)
        avg_r.append(adm_avg_r)
        avg_f1.append(adm_avg_f1)
        llprint("\rtest step: {} / {}".format(step, len(data_eval)))

    # ddi rate
    ddi_rate = ddi_rate_score(smm_record, path="../data/ready/ddi_A_final.pkl")

    llprint(
        "\nDDI Rate: {:.4}, Jaccard: {:.4},  PRAUC: {:.4}, AVG_PRC: {:.4}, AVG_RECALL: {:.4}, AVG_F1: {:.4}, AVG_MED: {:.4}\n".format(
            ddi_rate,
            np.mean(ja),
            np.mean(prauc),
            np.mean(avg_p),
            np.mean(avg_r),
            np.mean(avg_f1),
            med_cnt / visit_cnt,
        )
    )

    return (
        ddi_rate,
        np.mean(ja),
        np.mean(prauc),
        np.mean(avg_p),
        np.mean(avg_r),
        np.mean(avg_f1),
        med_cnt / visit_cnt,
    )
def generate_detailed_predictions(model, data, output_path, num_patients):
    model.eval()
    all_patients_data = []

    print(f"Starting to generate JSON report...")
    with torch.no_grad():
        for patient_idx, patient_visits in enumerate(data):
            patient_data = {
                "patient_id": patient_idx,
                "visits": []
            }
            for visit_idx, adm in enumerate(patient_visits):
                input_visits = patient_visits[: visit_idx + 1]
                target_output, _ = model(input_visits)
                actual_meds = [int(med_id) for med_id in adm[2]]
                pred_probs = torch.sigmoid(target_output).detach().cpu().numpy()[0]
                predicted_meds_with_probs = []
                topk = 131
                top_preds = heapq.nlargest(topk, enumerate(pred_probs), key=lambda x: x[1])  # [(med_id, prob), ...]
                for med_id, prob in top_preds:
                    predicted_meds_with_probs.append([int(med_id), float(prob)])
                visit_data = {
                    "visit_id": visit_idx,
                    "actual": actual_meds,
                    "predicted": predicted_meds_with_probs
                }
                patient_data["visits"].append(visit_data)
            all_patients_data.append(patient_data)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_patients_data, f, indent=2)
    print("Down!")

# ======================================================================

def main():

    # load data
    data_path = "../data/ready/records_final.pkl"
    voc_path = "../data/ready/voc_final.pkl"

    ddi_adj_path = "../data/ready/ddi_A_final.pkl"
    ddi_mask_path = "../data/ready/ddi_mask_H.pkl"
    molecule_path = "../data/ready/atc3toSMILES.pkl"
    device = torch.device("cuda:{}".format(args.cuda))

    ddi_adj = dill.load(open(ddi_adj_path, "rb"))
    ddi_mask_H = dill.load(open(ddi_mask_path, "rb"))
    data = dill.load(open(data_path, "rb"))
    molecule = dill.load(open(molecule_path, "rb"))

    voc = dill.load(open(voc_path, "rb"))
    diag_voc, pro_voc, med_voc = voc["diag_voc"], voc["pro_voc"], voc["med_voc"]

    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]
    eval_len = int(len(data[split_point:]) / 2)
    data_test = data[split_point : split_point + eval_len]
    data_eval = data[split_point + eval_len :]

    MPNNSet, N_fingerprint, average_projection = buildMPNN(
        molecule, med_voc.idx2word, 2, device
    )
    voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))

    model = SafeDrugModel(
        voc_size,
        ddi_adj,
        ddi_mask_H,
        MPNNSet,
        N_fingerprint,
        average_projection,
        emb_dim=args.dim,
        device=device,
    )

    if args.Test:
        model.load_state_dict(torch.load(open(args.resume_path, "rb")))
        model.to(device=device)
        if args.generate_json:
            generate_detailed_predictions(
                model=model,
                data=data_test,
                output_path=args.json_path,
                num_patients=args.num_patients
            )
        return

    model.to(device=device)
    # print('parameters', get_n_params(model))
    # exit()
    optimizer = Adam(list(model.parameters()), lr=args.lr)

    # start iterations
    history = defaultdict(list)
    best_epoch, best_ja = 0, 0

    EPOCH = 50
    for epoch in range(EPOCH):
        tic = time.time()
        print("\nepoch {} --------------------------".format(epoch + 1))

        model.train()
        for step, input in enumerate(data_train):

            loss = 0
            for idx, adm in enumerate(input):

                seq_input = input[: idx + 1]
                loss_bce_target = np.zeros((1, voc_size[2]))
                loss_bce_target[:, adm[2]] = 1

                loss_multi_target = np.full((1, voc_size[2]), -1)
                for idx_item, item in enumerate(adm[2]):
                    loss_multi_target[0][idx_item] = item

                result, loss_ddi = model(seq_input)

                loss_bce = F.binary_cross_entropy_with_logits(
                    result, torch.FloatTensor(loss_bce_target).to(device)
                )
                loss_multi = F.multilabel_margin_loss(
                    F.sigmoid(result), torch.LongTensor(loss_multi_target).to(device)
                )

                result = F.sigmoid(result).detach().cpu().numpy()[0]
                result[result >= 0.5] = 1
                result[result < 0.5] = 0
                y_label = np.where(result == 1)[0]
                current_ddi_rate = ddi_rate_score(
                    [[y_label]], path="../data/ready/ddi_A_final.pkl"
                )

                if current_ddi_rate <= args.target_ddi:
                    loss = 0.95 * loss_bce + 0.05 * loss_multi
                else:
                    beta = min(0, 1 + (args.target_ddi - current_ddi_rate) / args.kp) # bug fix: should be max(0, ..) but follow original
                    loss = (
                        beta * (0.95 * loss_bce + 0.05 * loss_multi)
                        + (1 - beta) * loss_ddi
                    )

                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

            llprint("\rtraining step: {} / {}".format(step, len(data_train)))

        print()
        tic2 = time.time()
        ddi_rate, ja, prauc, avg_p, avg_r, avg_f1, avg_med = eval(
            model, data_eval, voc_size, epoch
        )
        print(
            "training time: {}, test time: {}".format(
                time.time() - tic, time.time() - tic2
            )
        )

        history["ja"].append(ja)
        history["ddi_rate"].append(ddi_rate)
        history["avg_p"].append(avg_p)
        history["avg_r"].append(avg_r)
        history["avg_f1"].append(avg_f1)
        history["prauc"].append(prauc)
        history["med"].append(avg_med)

        if epoch >= 5:
            print(
                "ddi: {}, Med: {}, Ja: {}, F1: {}, PRAUC: {}".format(
                    np.mean(history["ddi_rate"][-5:]),
                    np.mean(history["med"][-5:]),
                    np.mean(history["ja"][-5:]),
                    np.mean(history["avg_f1"][-5:]),
                    np.mean(history["prauc"][-5:]),
                )
            )

        torch.save(
            model.state_dict(),
            open(
                os.path.join(
                    "saved",
                    args.model_name,
                    "Epoch_{}_TARGET_{:.2}_JA_{:.4}_DDI_{:.4}.model".format(
                        epoch, args.target_ddi, ja, ddi_rate
                    ),
                ),
                "wb",
            ),
        )

        if epoch != 0 and best_ja < ja:
            best_epoch = epoch
            best_ja = ja

        print("best_epoch: {}".format(best_epoch))

    dill.dump(
        history,
        open(
            os.path.join(
                "saved", args.model_name, "history_{}.pkl".format(args.model_name)
            ),
            "wb",
        ),
    )


if __name__ == "__main__":
    main()