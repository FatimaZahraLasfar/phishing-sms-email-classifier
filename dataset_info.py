import pandas as pd
from huggingface_hub import hf_hub_download

repo_id = "LHRS-UM-FERI/MENTHOS-dataset-spam"
files = {
    "train": "spam_train.csv",
    "validation": "spam_val.csv",
    "test": "spam_test.csv",
}

total_legit = total_phish = 0
for split, filename in files.items():
    path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
    df = pd.read_csv(path)
    n_legit = (df["label"] == 0).sum()
    n_phish = (df["label"] == 1).sum()
    total_legit += n_legit
    total_phish += n_phish
    print(f"{split:10s}  total={len(df):>6}  legit={n_legit:>6}  phish={n_phish:>6}")

print(f"\nTOTAL       total={total_legit + total_phish:>6}  legit={total_legit:>6}  phish={total_phish:>6}")