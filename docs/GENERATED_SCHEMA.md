# `generated.csv` schema

Core identity fields are `candidate_id`, `smiles`, `charged_smiles`, `formula`,
`family`, `synthon_a`, `synthon_b`, `method_id`, and `seed`.

Spectrum fields contain predicted `uvb_auc`, `uva_auc`, `lambda_c_nm`, their
uncertainties and lower confidence bounds, conditional band transmittances, and
`spectrum_json`. MOST fields contain ΔH, Wh/kg, log/linear half-life at 305 K,
uncertainties, and confidence bounds. Safety fields include Kp,
phototoxicity/uncertainty, SA proxy, psoralen, known-structure and reactive
alerts. `similarity_D_A`/`similarity_D_B` and `ad_spectral`/`ad_most` expose the
two applicability domains.

`joint_uv_pass`, `most_pass`, `safety_pass`, and `joint_pass` are transparent
gate outputs. `reward_components_json` and `stage_rewards_json` preserve dense
scores. `failure_reasons` explains the decision. `selected` is fail-closed and
requires independent evaluator plus physical-oracle completion.
`not_iso_certified` is always true.

