# BT4012 – Fraudulent Bitcoin transaction detection (e0958591)

## Files
Put these three files in the folder you run the notebook from (the notebook reads them by these names):

| File | Contents |
|---|---|
| `train.csv` | Labelled transactions used for training. Each row is one transaction. Some transactions have no label (unknown); the notebook keeps only labelled rows. |
| `test.csv` | Transactions to generate predictions for. |
| `txs_edgelist.csv` | The transaction graph. Each row is a directed edge (`txId1`, `txId2`): a payment flow from `txId1` to `txId2`. Used for graph features and the graph model. |

## Run
```
pip install -r requirements.txt
jupyter notebook e0958591.ipynb      # Run All; takes tens of minutes on a laptop CPU (the neural-network cells are the slowest)
```
Outputs written to the same folder:
- `final_submission.csv` – final submission (first half of the columns removed in the second view + novelty nudge)
- `backup_submission.csv` – backup submission (same, without the novelty nudge)

## Approach
1. **Features:** the 165 anonymised features, plus graph features from the full edge list (in/out degree, in/out ratio,
   component size, 2-hop reach) and neighbour means of the first 20 features.
2. **Time axis:** each connected component of the edge list is one time step. The `time_step` column is used only to build
   rolling forward-in-time validation (train early, validate later) and for diagnostics; it is not a model input.
3. **Models:** logistic regression, random forest, extra trees, LightGBM, XGBoost, CatBoost, a residual MLP (seed-averaged,
   short training) and GraphSAGE. Class imbalance is handled with class weights.
4. **Base blend:** rank-average of ResMLP, LightGBM, XGBoost, CatBoost, ExtraTrees and GraphSAGE
   (`Blend_ResMLP+LGB+XGB+Cat+ET+GNN`).
5. **Late-regime detection (no labels):** a step is flagged when its alarm rate (share of its rows in the global top 5% of
   blend scores) falls below 25% of the median of earlier steps. This marks steps 43-49 as "late".
6. **Self-training on the late rows:** within each late step, the top 20% of rows by blend rank become pseudo-positives and
   the bottom 72% pseudo-negatives. LightGBM and XGBoost are trained on train + these pseudo-labelled rows, cross-fitted over
   two random halves, and re-score the late rows. Two column sets are used and rank-averaged: all columns, and all columns
   except the first half of the raw features (feat_1..feat_82). The new late ordering is mapped back onto the blend's own
   late score values, so early steps are untouched.
7. **Novelty nudge (final only):** the pseudo-label ranking is the blend's within-step rank plus 0.15 times a label-free
   novelty rank (mean distance to the 10 nearest licit training rows in a 30-component PCA space), re-ranked before the cut.
   The backup submission is the same pipeline without this nudge.

## Results (public leaderboard AUC)
| Stage | Public AUC |
|---|---|
| RF+ET+LGB+XGB+Cat blend | 0.95029 |
| ResMLP + trees + GraphSAGE blend | 0.95593 |
| + late-regime self-training (20% / 72%) | 0.96262 |
| + novelty nudge | 0.96611 |
