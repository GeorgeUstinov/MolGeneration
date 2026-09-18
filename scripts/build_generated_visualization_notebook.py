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

Notebook читает свежий balanced top-1000 из
`generated_multiclass_photoswitch.csv`, полученный одним conditional
SELFIES-GRU с family-токенами `<AZO>`, `<STILBENE>`, `<SPIROPYRAN>` и
`<DIARYLETHENE>`.

В выборке четыре сбалансированных класса: azo, stilbene, spiropyran и
diarylethene. Два последних имеют reaction-constrained open/closed pairs.
Второе химическое пространство — экспериментально обученный phototoxicity
screen; ADME-like величины используются только как признаки модели.
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
METRICS_PATH = ROOT / "outputs" / "uv_most_multiclass" / "generation_metrics.csv"
TOXICITY_METRICS_PATH = ROOT / "outputs" / "uv_most_multiclass" / "phototoxicity_model_metrics.csv"
TOXICITY_TRAINING_PATH = ROOT / "outputs" / "uv_most_multiclass" / "phototoxicity_training.csv"
df = pd.read_csv(CSV_PATH)
required = {
    "selection_rank", "candidate_id", "rank_within_family", "family",
    "canonical_smiles", "state_A_smiles", "state_B_smiles",
    "pair_valid", "eval_uv_lambda_nm", "eval_uv_uncertainty_nm", "sa_score",
    "reward", "selection_score", "target_A_pass", "target_B_pass",
    "photoswitch_space_pass", "toxicity_space_pass", "phototoxic_probability",
    "phototoxic_uncertainty", "phototoxic_upper_confidence", "toxicity_similarity",
    "toxicity_in_domain", "toxicity_pass",
    "joint_success", "PSS_pred", "quantum_yield_pred", "deltaH_kJ_mol",
    "stored_energy_MJ_kg", "final_status",
}
missing = required.difference(df.columns)
if missing:
    raise KeyError(f"generated_multiclass_photoswitch.csv is missing: {sorted(missing)}")
if len(df) != 1000 or df["canonical_smiles"].nunique() != 1000:
    raise ValueError("Expected exactly 1000 globally unique fresh candidates")
print(f"Loaded {len(df):,} rows from {CSV_PATH}")
print("Families:", df["family"].value_counts().to_dict())
print("Unique structures:", df.groupby("family")["canonical_smiles"].nunique().to_dict())
print("Selection ranks:", (int(df["selection_rank"].min()), int(df["selection_rank"].max())))
        """
    ),
    md("## Состав свежего top-1000 и ранжирование"),
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
          median_phototoxic_probability=("phototoxic_probability", "median"),
          toxicity_in_domain_rate=("toxicity_in_domain", "mean"),
          median_selection_score=("selection_score", "median"),
      )
)
display(summary)
display(pd.crosstab(df["family"], df["final_status"]))

family_order = ["azo", "stilbene", "spiropyran", "diarylethene"]
colors = {"azo":"tab:orange", "stilbene":"tab:blue", "spiropyran":"tab:green", "diarylethene":"tab:purple"}
fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
df["family"].value_counts().reindex(family_order).plot.bar(
    ax=axes[0], color=[colors[family] for family in family_order]
)
axes[0].set(title="Balanced shortlist", xlabel="family", ylabel="candidates")
for family, part in df.groupby("family"):
    axes[1].hist(part["selection_score"], bins=28, alpha=0.6, color=colors[family], label=family)
    axes[2].scatter(
        part["rank_within_family"], part["selection_score"],
        s=11, alpha=0.45, color=colors[family], label=family,
    )
axes[1].set(title="Evaluator/proxy score", xlabel="selection score", ylabel="count")
axes[2].set(title="Score along family rank", xlabel="rank within family", ylabel="selection score")
for axis in axes[1:]:
    axis.legend()
for axis in axes:
    axis.grid(alpha=0.2)
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
    c=azo["selection_score"], cmap="viridis", s=16, alpha=0.5,
)
axes[0].axvline(20, color="black", ls="--")
axes[0].axhspan(4, 24, color="tab:orange", alpha=0.1)
axes[0].set_yscale("log")
axes[0].set(xlabel="Predicted |λE−λZ| (nm)", ylabel="Evaluator half-life (h)", title="Azo proxy space")
fig.colorbar(scatter, ax=axes[0], label="selection score")
axes[1].hist(azo["pred_delta_lambda_nm"], bins=30, color="tab:purple", alpha=0.75)
axes[1].axvline(20, color="black", ls="--")
axes[1].set(xlabel="Predicted E/Z separation (nm)", ylabel="count", title="Azo spectral separation")
for axis in axes:
    axis.grid(alpha=0.2)
plt.tight_layout()
plt.show()
        """
    ),
    md("## Экспериментально обученный phototoxicity screen"),
    code(
        """
fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
df["toxicity_risk_band"] = pd.cut(
    df["phototoxic_upper_confidence"], bins=[-np.inf,.35,.70,np.inf],
    labels=["low_model_risk", "uncertain", "screened_out_high_risk"], right=False,
)
for family, part in df.groupby("family"):
    axes[0].hist(part["phototoxic_probability"], bins=25, alpha=.5, color=colors[family], label=family)
    axes[1].scatter(part["toxicity_similarity"], part["phototoxic_probability"], s=10, alpha=.3,
                    color=colors[family], label=family)
    axes[2].hist(part["phototoxic_upper_confidence"], bins=25, alpha=.5, color=colors[family], label=family)
axes[0].set(title="Predicted phototoxicity", xlabel="probability", ylabel="count")
axes[1].axvline(.15, color="black", ls="--", label="AD threshold")
axes[1].set(title="Applicability domain", xlabel="max Tanimoto to U16", ylabel="probability")
axes[2].axvline(.70, color="red", ls="--", label="screen threshold")
axes[2].set(title="Upper-confidence screen", xlabel="p + 1.645·ensemble std", ylabel="count")
for axis in axes:
    axis.grid(alpha=.2); axis.legend(fontsize=8)
plt.tight_layout(); plt.show()

display(df.groupby("family").agg(
    screen_pass_rate=("toxicity_pass","mean"),
    in_domain_rate=("toxicity_in_domain","mean"),
    median_probability=("phototoxic_probability","median"),
    median_upper_confidence=("phototoxic_upper_confidence","median"),
))
display(pd.crosstab(df["family"], df["toxicity_risk_band"], dropna=False))
print("Важно: screen_pass — критерий приоритизации, а не экспериментальное доказательство безопасности.")
        """
    ),
    md("## Лучшие пары состояний каждого класса"),
    code(
        """
def show_family_pairs(family, count=9):
    top = (
        df[df["family"].eq(family)]
        .sort_values("rank_within_family")
        .drop_duplicates("canonical_smiles")
        .head(count)
    )
    display(top[[
        "candidate_id", "rank_within_family", "canonical_smiles",
        "state_A_smiles", "state_B_smiles", "eval_uv_lambda_nm", "sa_score",
        "selection_score", "final_status",
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
show_family_pairs("spiropyran")
show_family_pairs("diarylethene")
        """
    ),
    md(
        """
### Научная граница результата

Все классы входят в общий conditional checkpoint, но connectivity-changing
spiropyran и diarylethene pairs дополнительно ограничены реакционными шаблонами.
Фототоксичность обучена на 53 структурах QDB и поэтому для новых scaffold часто
находится вне applicability domain. PSS, quantum yield, ΔH и stored energy
намеренно остаются пустыми до физической валидации.
        """
    ),
    md(
        """
## Метрики генерации

Отдельный финальный вывод показывает метрики как для всех proposals,
так и для отобранного top-1000. Единичные метрики top-1000 являются следствием
целевого отбора; качество самого generator следует оценивать по scope
`all_proposals`.
        """
    ),
    code(
        """
metrics = pd.read_csv(METRICS_PATH)
required_metrics = {
    "scope", "family", "n_generated", "n_valid", "n_unique",
    "n_not_in_training", "n_joint_success", "validity", "uniqueness",
    "novelty", "joint_success_rate", "diversity",
}
missing_metrics = required_metrics.difference(metrics.columns)
if missing_metrics:
    raise KeyError(f"generation_metrics.csv is missing: {sorted(missing_metrics)}")

scope_order = ["all_proposals", "top_1000"]
family_order = ["all", "azo", "stilbene", "spiropyran", "diarylethene"]
metrics["scope"] = pd.Categorical(metrics["scope"], scope_order, ordered=True)
metrics["family"] = pd.Categorical(metrics["family"], family_order, ordered=True)
metrics = metrics.sort_values(["scope", "family"]).reset_index(drop=True)

formulas = pd.DataFrame({
    "metric": ["Validity", "Uniqueness", "Novelty", "JSR", "Diversity"],
    "definition": [
        "Nvalid / Ngenerated",
        "Nunique_valid / Nvalid",
        "Nunique_valid_not_in_training / Nunique_valid",
        "N(target_A_pass AND target_B_pass) / Ngenerated",
        "1 - mean pairwise Morgan/Tanimoto similarity over unique valid structures",
    ],
})
display(formulas)

metrics_display = metrics.copy()
rate_columns = ["validity", "uniqueness", "novelty", "joint_success_rate", "diversity"]
metrics_display[rate_columns] = metrics_display[rate_columns].round(6)
display(metrics_display)

overall = (
    metrics[metrics["family"].eq("all")]
    .set_index("scope")[rate_columns]
    .rename(columns={"joint_success_rate": "JSR"})
)
axes = overall.T.plot.bar(figsize=(12, 4.8), ylim=(0, 1.05), width=0.78)
axes.set(title="Generation metrics: raw proposals vs selected top-1000", xlabel="metric", ylabel="value")
axes.grid(axis="y", alpha=0.25)
axes.legend(title="scope", loc="lower right")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

toxicity_model_metrics = pd.read_csv(TOXICITY_METRICS_PATH)
toxicity_training = pd.read_csv(TOXICITY_TRAINING_PATH)
linked = toxicity_training.dropna(subset=["photochem_label"])
toxicity_evidence = pd.DataFrame([{
    "QDB_structured_training_rows": len(toxicity_training),
    "QDB_phototoxic": int(toxicity_training["phototoxic"].sum()),
    "QDB_non_phototoxic": int((toxicity_training["phototoxic"] == 0).sum()),
    "PhotoChem_CAS_linked_rows": len(linked),
    "PhotoChem_agreement": linked["photochem_agrees"].astype(str).str.lower().eq("true").mean(),
}])
display(toxicity_model_metrics.round(6))
display(toxicity_evidence.round(6))
        """
    ),
]

notebook = nbf.v4.new_notebook(cells=cells)
notebook.metadata.kernelspec = {"display_name": "Python 3 (uvgen)", "language": "python", "name": "python3"}
notebook.metadata.language_info = {"name": "python", "version": "3.12"}
NOTEBOOK.write_text(nbf.writes(notebook), encoding="utf-8")
print(f"Wrote {NOTEBOOK}")
