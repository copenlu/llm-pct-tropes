import pandas as pd
import argparse
import torch
import random
import numpy as np
from torch.utils.data import Dataset
from transformers import pipeline
from tqdm import tqdm
import ipdb
from rouge_score import rouge_scorer


def enforce_reproducibility(seed=1000):
    # Sets seed manually for both CPU and CUDA
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For atomic operations there is currently
    # no simple way to enforce determinism, as
    # the order of parallel operations is not known.
    # CUDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # System based
    random.seed(seed)
    np.random.seed(seed)


class NLIDataset(Dataset):

    def __init__(self, dframe):
        self.dframe = dframe

    def __len__(self):
        return len(self.dframe)

    def __getitem__(self, item):
        row = self.dframe.iloc[item]
        return {
            "text_pair": row[trope_column],
            "text": row['sentences']
        }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--tropes_csv", type=str, help="CSV with the tropes", required=True)
    parser.add_argument("--seed", type=int, help="Random seed", default=1000)


    args = parser.parse_args()

    trope_column = 'distilled_trope'

    enforce_reproducibility(args.seed)
    dframe = pd.read_csv(args.tropes_csv).fillna('')

    dset = NLIDataset(dframe)

    device = 'cpu'
    if torch.backends.mps.is_available():
        print("Using MPS")
        device = 'cpu'
    elif torch.cuda.is_available():
        print("Using CUDA")
        device = 'cuda'

    pipe = pipeline(
        "text-classification",
        model='ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli',
        device=device
    )

    out_preds = []

    # Step 1, NLI
    for out in tqdm(pipe(dset, batch_size=32), total=len(dset)):
        out_preds.append(out['label'])
    out_preds = np.array(out_preds)
    print(out_preds)
    for lab in ['entailment', 'neutral', 'contradiction']:
        print(f"{lab}: {(out_preds == lab).mean()}")

    dframe.insert(len(dframe.columns), "NLI", out_preds)

    # Step 2, ROUGE scores
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    r1_scores = []
    r2_scores = []
    rl_scores = []
    for j,row in tqdm(dframe.iterrows(), total=len(dframe)):
        scores = scorer.score(row[trope_column], row['sentences'])
        r1_scores.append(scores['rouge1'].fmeasure)
        r2_scores.append(scores['rouge2'].fmeasure)
        rl_scores.append(scores['rougeL'].fmeasure)


    print(f"Rouge1: {np.array(r1_scores).mean()}")
    print(f"Rouge2: {np.array(r2_scores).mean()}")
    print(f"RougeL: {np.array(rl_scores).mean()}")

    dframe.insert(len(dframe.columns), "Rouge1", r1_scores)
    dframe.insert(len(dframe.columns), "Rouge2", r2_scores)
    dframe.insert(len(dframe.columns), "RougeL", rl_scores)


    dframe.to_csv(f"{args.tropes_csv}_intrinsic_eval.csv", index=None)