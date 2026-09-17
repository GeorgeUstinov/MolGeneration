from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "Generated_Molecules_Visualization.ipynb"


def md(source: str):
    return nbf.v4.new_markdown_cell(source.strip())


def code(source: str):
    return nbf.v4.new_code_cell(source.strip())


cells = [
    md(
        """
# Визуализация единой multiclass photoswitch-модели

Notebook читает `generated_multiclass_photoswitch.csv`, полученный одним
conditional SELFIES-GRU с family-токенами `<AZO>` и `<STILBENE>`.

Оба класса имеют явные E/Z-пары и общий UV proxy. Azo дополнительно имеет
обучаемые half-life и E/Z spectral-separation proxies. Для stilbene-like
half-life, PSS, quantum yield, ΔH и stored energy отсутствуют и не подменяются.
        """
    ),
    code(
        """
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw

RDLogger.DisableLog("rdApp.*")
ROOT = Path.cwd().resolve()
CSV_PATH = ROOT / "generated_multiclass_photoswitch.csv"
df = pd.read_csv(CSV_PATH)
required = {
    "family", "canonical_smiles", "state_A_smiles", "state_B_smiles",
    "pair_valid", "eval_uv_lambda_nm", "eval_uv_uncertainty_nm", "sa_score",
    "reward", "PSS_pred", "quantum_yield_pred", "deltaH_kJ_mol",
    "stored_energy_MJ_kg", "final_status",
}
missing = required.difference(df.columns)
if missing:
    raise KeyError(f"generated_multiclass_photoswitch.csv is missing: {sorted(missing)}")
print(f"Loaded {len(df):,} rows from {CSV_PATH}")
print("Families:", df["family"].value_counts().to_dict())
print("Unique structures:", df.groupby("family")["canonical_smiles"].nunique().to_dict())
        """
    ),
    md("## Размеры классов и итоговые proxy-status"),
    code(
        """
summary = (
    df.groupby("family", as_index=False)
      .agg(
          rows=("canonical_smiles", "size"),
          unique=("canonical_smiles", "nunique"),
          proxy_pass=("final_status", lambda x: int(x.str.startswith("PASS").sum())),
          median_uv_nm=("eval_uv_lambda_nm", "median"),
          median_uv_uncertainty_nm=("eval_uv_uncertainty_nm", "median"),
          median_sa=("sa_score", "median"),
      )
)
display(summary)
display(pd.crosstab(df["family"], df["final_status"]))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
df["family"].value_counts().plot.bar(ax=axes[0], color=["tab:orange", "tab:blue"])
axes[0].set(title="Generated valid pairs", xlabel="family", ylabel="rows")
status_table = pd.crosstab(df["family"], df["final_status"])
status_table.plot.bar(stacked=True, ax=axes[1])
axes[1].set(title="Proxy status by family", xlabel="family", ylabel="rows")
plt.tight_layout()
plt.show()
        """
    ),
    md("## Общий UV proxy и synthetic accessibility"),
    code(
        """
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for family, part in df.groupby("family"):
    axes[0].hist(part["eval_uv_lambda_nm"], bins=30, alpha=0.55, label=family)
    axes[1].scatter(part["eval_uv_lambda_nm"], part["eval_uv_uncertainty_nm"], s=9, alpha=0.25, label=family)
    axes[2].hist(part["sa_score"], bins=30, alpha=0.55, label=family)
axes[0].axvspan(300, 400, color="green", alpha=0.08)
axes[0].set(xlabel="Independent evaluator λmax (nm)", ylabel="count", title="UV prediction")
axes[1].set(xlabel="λmax (nm)", ylabel="uncertainty (nm)", title="UV uncertainty")
axes[2].axvline(6.0, color="red", ls="--")
axes[2].set(xlabel="RDKit SA Score", ylabel="count", title="Synthesis proxy")
for axis in axes:
    axis.legend()
    axis.grid(alpha=0.2)
plt.tight_layout()
plt.show()
        """
    ),
    md("## Azo-specific spectral separation и half-life"),
    code(
        """
azo = df[df["family"].eq("azo")].dropna(subset=["pred_delta_lambda_nm", "eval_half_life_h"])
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
scatter = axes[0].scatter(
    azo["pred_delta_lambda_nm"], azo["eval_half_life_h"],
    c=azo["reward"], cmap="viridis", s=12, alpha=0.45,
)
axes[0].axvline(20, color="black", ls="--")
axes[0].axhspan(4, 24, color="tab:orange", alpha=0.1)
axes[0].set_yscale("log")
axes[0].set(xlabel="Predicted |λE−λZ| (nm)", ylabel="Evaluator half-life (h)", title="Azo proxy space")
fig.colorbar(scatter, ax=axes[0], label="reward")
axes[1].hist(azo["pred_delta_lambda_nm"], bins=30, color="tab:purple", alpha=0.75)
axes[1].axvline(20, color="black", ls="--")
axes[1].set(xlabel="Predicted E/Z separation (nm)", ylabel="count", title="Azo spectral separation")
for axis in axes:
    axis.grid(alpha=0.2)
plt.tight_layout()
plt.show()
        """
    ),
    md("## Лучшие E/Z-пары каждого класса"),
    code(
        """
def show_family_pairs(family, count=9):
    top = (
        df[df["family"].eq(family)]
        .sort_values(["final_status", "reward"], ascending=[False, False])
        .drop_duplicates("canonical_smiles")
        .head(count)
    )
    display(top[[
        "canonical_smiles", "state_A_smiles", "state_B_smiles",
        "eval_uv_lambda_nm", "sa_score", "reward", "final_status",
    ]])
    state_a = [Chem.MolFromSmiles(s) for s in top["state_A_smiles"]]
    state_b = [Chem.MolFromSmiles(s) for s in top["state_B_smiles"]]
    legends_a = [f"{family} A | UV={r.eval_uv_lambda_nm:.0f} | SA={r.sa_score:.2f}" for r in top.itertuples()]
    legends_b = [f"{family} B | UV={r.eval_uv_lambda_nm:.0f} | SA={r.sa_score:.2f}" for r in top.itertuples()]
    print(f"{family}: state A")
    display(Draw.MolsToGridImage(state_a, legends=legends_a, molsPerRow=3, subImgSize=(320, 250), useSVG=True))
    print(f"{family}: corresponding state B")
    display(Draw.MolsToGridImage(state_b, legends=legends_b, molsPerRow=3, subImgSize=(320, 250), useSVG=True))

show_family_pairs("azo")
show_family_pairs("stilbene")
        """
    ),
    md(
        """
### Научная граница результата

Оба класса генерируются одним checkpoint. Для azo доступны M01-derived half-life
и E/Z spectral proxies. Для stilbene-like подтверждены лишь diaryl/heteroaryl
alkene E/Z pair и общий UVVisML proxy. PSS, quantum yield, ΔH и stored energy
намеренно остаются пустыми для обоих классов до физической валидации.
        """
    ),
]

notebook = nbf.v4.new_notebook(cells=cells)
notebook.metadata.kernelspec = {"display_name": "Python 3 (uvgen)", "language": "python", "name": "python3"}
notebook.metadata.language_info = {"name": "python", "version": "3.12"}
NOTEBOOK.write_text(nbf.writes(notebook), encoding="utf-8")
print(f"Wrote {NOTEBOOK}")
