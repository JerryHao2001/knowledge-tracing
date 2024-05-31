import os
import argparse
import json
import pandas as pd

import torch


from data_loaders.assist2009 import ASSIST2009
from data_loaders.assist2015 import ASSIST2015
from data_loaders.algebra2005 import Algebra2005
from data_loaders.statics2011 import Statics2011

from models.dkt import DKT
from models.dkt_plus import DKTPlus
from models.dkvmn import DKVMN
from models.sakt import SAKT
from models.gkt import PAM, MHA


def main(model_name, dataset_name, student_hist_path):

    ckpt_path = os.path.join("ckpts", model_name, dataset_name)

    if not os.path.isdir(ckpt_path):
        print("Trained model doesn't exist")
        return

    with open(os.path.join(ckpt_path,"model_config.json")) as f:
        model_config = json.load(f)

    with open(os.path.join(ckpt_path,"train_config.json")) as f:
        train_config = json.load(f)

    seq_len = train_config["seq_len"]

    if dataset_name == "ASSIST2009":
        dataset = ASSIST2009(seq_len)
    elif dataset_name == "ASSIST2015":
        dataset = ASSIST2015(seq_len)
    elif dataset_name == "Algebra2005":
        dataset = Algebra2005(seq_len)
    elif dataset_name == "Statics2011":
        dataset = Statics2011(seq_len)

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    if model_name == "dkt":
        model = DKT(**model_config).to(device)
    elif model_name == "dkt+":
        model = DKTPlus(**model_config).to(device)
    elif model_name == "dkvmn":
        model = DKVMN(**model_config).to(device)
    elif model_name == "sakt":
        model = SAKT(**model_config).to(device)
    elif model_name == "gkt":
        if model_config["method"] == "PAM":
            model = PAM(**model_config).to(device)
        elif model_config["method"] == "MHA":
            model = MHA(**model_config).to(device)
    else:
        print("The wrong model name was used...")
        return

    model.load_state_dict(torch.load(os.path.join(ckpt_path, "model.ckpt")))

    exer_hist = pd.read_csv(student_hist_path)
    exer_hist["qidx"] = exer_hist.q.map(dataset.q2idx)
    q = torch.tensor(exer_hist.qidx,dtype=torch.long,device=device)
    r = torch.tensor(exer_hist.r,dtype=torch.long,device=device)  

    with torch.no_grad():
        model.eval()
        y = model(q, r)

    report = pd.DataFrame(list(dataset.q2idx.keys()), columns=['Skills'])
    report["Correct_Probability"] = y[-1].cpu().numpy()
    report.to_csv(student_hist_path[:-4]+"_report.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="dkt",
        help="The name of the model to train. \
            The possible models are in [dkt, dkt+, dkvmn, sakt, gkt]. \
            The default model is dkt."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="ASSIST2009",
        help="The name of the dataset to use in training. \
            The possible datasets are in \
            [ASSIST2009, ASSIST2015, Algebra2005, Statics2011]. \
            The default dataset is ASSIST2009."
    )
    parser.add_argument(
        "--student_hist_path",
        type=str,
        default="student001.csv",
        help="The path of the csv recording the   \
            historical performances of a student \
            used for inference."
    )

    args = parser.parse_args()

    main(args.model_name, args.dataset_name, args.student_hist_path)