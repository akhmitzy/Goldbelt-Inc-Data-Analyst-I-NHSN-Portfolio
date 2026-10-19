# Notebook 00_01: Synthetic Data Generator + Data Quality Validation Pipeline

The notebook **00_01_synthetic_data_and_data_quality.ipynb** is ready. I executed it end to end in Jupyter with no errors, so the saved file includes its outputs and plots. It should run top to bottom on a free Colab CPU runtime in a few minutes, but I couldn't test it in Colab itself.

## What it contains (32 cells)

- **Generator:** 160 synthetic facilities over 24 months, with patient-month denominators by access type and an event line list. Event rates are overdispersed, and a simulated CVC-reduction collaborative is built in for the later notebooks.
- **Planted errors with ground truth:** ten error types, including missing months, x10 typos, duplicates, bad dates, and over-reporting. This lets every method be scored against known truth.
- **Validation:** ten documented rules, each with an ID, severity, proposed system modification and form-instruction edit. Results are reported per rule.
- **Statistics:** robust z-scores, Poisson plausibility, and a funnel plot with an overdispersion adjustment.
- **Neural net:** a PyTorch denoising autoencoder compared with Isolation Forest, the rules and a hybrid list.
- **Outputs:** facility scorecard with outreach tiers, a prioritized modification log, a data dictionary, CSVs and a multi-sheet Excel workbook.

## What the results show (seed 42, so your numbers may differ slightly)

- The rules catch about 100% of the error types they were written for, with no false alarms on clean events.
- The rules almost never catch over-reporting (13% recall), because each extra event looks valid. Only the Poisson check (52%) and Isolation Forest (46%) pick up much of it, and the hybrid reaches 56%.
- The autoencoder came out comparable to Isolation Forest (AUROC 0.85 vs 0.94), not clearly better, and I kept that in the write-up.

## Before you publish

- **Autoencoder tuning:** my first autoencoder (wider, 120 epochs) scored poorly because it learned to reproduce the outliers. I chose the final settings on two *other* simulation seeds, not the evaluation seed, and the notebook says so.
- **Rule definitions:** they are illustrative, including the 21-day duplicate window and the 30-day lateness cutoff. Check them against the current NHSN Dialysis Event Protocol before presenting them as real.
- **Scorecard cut-offs and weights:** these are my own proposals, not validated standards.
- **Colab badge:** replace `YOUR_GITHUB_USERNAME` in the first cell with your GitHub username.

## Next step

Which should come next: notebook 02 (the negative binomial and Bayesian hierarchical rate model) or notebook 03 (interrupted time series and difference-in-differences)? Both reuse the same dataset.
