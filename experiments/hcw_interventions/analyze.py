"""results/<scenario>/all_runs.tsv -> outputs/{summary.csv, hci.png, hce.png}"""
from pathlib import Path

import matplotlib.pyplot as plt

from hcw import load, metrics, plots

here = Path(__file__).parent
out = here / "outputs"
out.mkdir(exist_ok=True)

df = load.load_all(here / "results")
runs = metrics.per_run(df)
runs.to_csv(out / "per_run.csv", index=False)
metrics.summarize(runs).to_csv(out / "summary.csv")

for col, title in [("HCI", "HCWs infected or quarantined [%]"), ("HCE", "new HCW infections per day")]:
    plt.figure(figsize=(8, 4))
    plots.band(df, col)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out / f"{col.lower()}.png", dpi=150)
    plt.close()
