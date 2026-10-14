# Portfolio Proposal: NHSN Dialysis Event Surveillance and Prevention Analytics

**Title:** *From Raw Surveillance Data to Prevention Action: Data Quality, Risk Modeling, and Outreach Analytics for Outpatient Hemodialysis*

**Target role:** Data Analyst I, NHSN Dialysis Event Prevention and Surveillance Training Support (Goldbelt Professional Services, Surveillance Branch)

**Stack:** Python, Google Colab, PyTorch, PyMC, statsmodels, scikit-learn

Each project maps to a duty in the job description, so a hiring manager can see the connection directly. Everything runs in Google Colab with Python.

---

## Data Strategy

NHSN data is not publicly available at patient level, so the portfolio uses two layers:

1. **Synthetic NHSN-like dataset (core).** Build a generator that mimics the Dialysis Event module:
   - Facilities, with monthly patient-months as the denominator.
   - Vascular access type (AV fistula, AV graft, tunneled CVC, nontunneled CVC).
   - Event types: positive blood culture, IV antimicrobial start, pus/redness/swelling at the access site.
   - Planted data-quality problems: missing months, duplicate events, implausible dates, denominator/numerator mismatches, late reporting.
2. **Public facility-level data (validation layer).** CMS Care Compare dialysis facility files from data.cms.gov, for realistic facility characteristics and benchmarks. Check current column names before using them.
3. **Data statement.** State clearly in the README that no PHI or real NHSN data was used.

---

## Project 1: Data Quality and Validation Pipeline

*Maps to: data management and cleaning, data validation projects, system modification recommendations.*

- **Rule-based validation:**
  - dates outside the reporting month
  - events with no denominator
  - duplicate events within a 21-day window
  - access-type inconsistencies
  - outlier patient-month counts
- **Statistical anomaly detection:** robust z-scores (median/MAD) and funnel plots with control limits per facility.
- **Neural net component:** a denoising autoencoder (PyTorch) on facility-month feature vectors, where reconstruction error flags suspicious submissions. Compare it against Isolation Forest and the rule-based flags, with precision/recall against the planted errors.
- **Deliverables:**
  - a data quality scorecard per facility
  - a tracking log of proposed "system modifications" (issue, impact, priority)
  - recommended edits to the form instructions

---

## Project 2: Rate Estimation and Risk Adjustment

*Maps to: NHSN analyses for CMS quality incentive programs, user reports.*

- Compute the standard metrics: events per 100 patient-months, stratified by access type.
- **Negative binomial and zero-inflated Poisson GLMs** with a log(patient-months) offset, tested for overdispersion.
- **Bayesian hierarchical model (PyMC):** facility random effects nested within state or ESRD Network, with partial pooling and shrinkage estimates. This stabilizes small-facility rates and is the most defensible way to avoid mislabeling small facilities as outliers.
- **Standardized infection ratios (SIR)** with exact Poisson confidence intervals, compared with posterior-based ratios.
- **Diagnostics:** posterior predictive checks, LOO-CV, calibration.

---

## Project 3: Trend Analysis and Intervention Evaluation

*Maps to: prevention efforts, advocacy with ESRD Networks, evaluating outreach.*

- **Interrupted time series** (segmented regression with autocorrelation handling) evaluating a simulated prevention collaborative, such as a CVC-reduction campaign.
- **Difference-in-differences** with facility fixed effects, plus an event-study plot to check parallel trends.
- **Forecasting** with SARIMAX/Prophet and a small LSTM, benchmarked by rolling-origin cross-validation. Report honestly whether the neural net beats the simple baseline.
- **Early-warning system:** a CUSUM or EWMA control chart that flags facilities with rising rates, plus a Poisson scan-style cluster check by Network or state.

---

## Project 4: Predictive Modeling for Outreach Prioritization

*Maps to: user outreach, QI organizations, health departments, enhancing usability of surveillance data.*

- Predict which facilities are likely to have high rates (or poor data quality) next quarter.
- **Models:** regularized logistic regression, gradient boosting, and a **tabular neural network** (PyTorch MLP with embeddings and dropout). Use temporal train/test splits to avoid leakage.
- **Evaluation:** AUROC, AUPRC, calibration curves, decision-curve analysis. Compare models with bootstrap confidence intervals.
- **Explainability:** SHAP values and partial dependence, since public health users need to know why a facility was flagged.
- **Fairness check:** performance by facility size, ownership, and rural/urban status, so the model does not systematically under-flag small or rural facilities.

---

## Project 5: Communication, Training, and Automation

*Maps to: 20-30 trainings a year, webpage content, inquiry handling, reports.*

- **Inquiry triage:** synthetic user questions classified by topic (enrollment, event definitions, denominator entry, CMS reporting) using TF-IDF + logistic regression versus a small transformer embedding model. Forecast monthly inquiry volume (the posting expects about 212 per month, range 122-415) for staffing.
- **Automated facility report generator:** one-page PDF/HTML per facility with rates, trends, benchmark, and data quality score in plain language.
- **Training deck and one-page "how to read your rates" handout** for dialysis providers.
- **Draft Dialysis Safety webpage text** on CVC reduction and bloodstream infection prevention.

---

## Repository Structure

```
nhsn-dialysis-analytics/
├── README.md                  # problem, data statement, key findings, limitations
├── notebooks/
│   ├── 00_synthetic_data.ipynb
│   ├── 01_data_quality.ipynb
│   ├── 02_rates_bayesian_glm.ipynb
│   ├── 03_trends_its_did_forecast.ipynb
│   ├── 04_prediction_nn_shap.ipynb
│   └── 05_reports_nlp_training.ipynb
├── src/                       # reusable functions (validation, metrics, plotting)
├── reports/                   # sample facility report, SOP, data dictionary
├── docs/                      # protocol-style methods, change log
└── requirements.txt
```

Each notebook should have an "Open in Colab" badge, fixed random seeds, and a short "So what for public health?" section.

---

## What Makes It Read as Serious to to Goldbelt Inc

- **Domain fidelity:** correct NHSN terminology, denominators in patient-months, and an acknowledgment that these are events, not necessarily infections.
- **Documentation like a government contractor:** data dictionary, SOP for validation, change log, version control, and a limitations section. The role mentions government procedures and tracked modifications.
- **Honest model comparisons:** show where simple methods win. Overclaiming on neural nets is a red flag for public health reviewers.
- **Plain-language outputs** for non-statisticians, aligned with the outreach and training duties.
- **Tooling parity:** the posting names Excel, Access, SAS, and R. Add a short appendix that reproduces one analysis in SAS-style PROC logic (or R) and exports a clean Excel report, to show you are not Python-only.

---

## Suggested 4-Week Plan

| Week | Focus |
|---|---|
| 1 | Synthetic generator, data dictionary, Project 1 |
| 2 | Project 2 (GLMs and Bayesian hierarchical model) |
| 3 | Projects 3 and 4 (ITS/DiD, forecasting, NN, SHAP) |
| 4 | Project 5, README polish, facility report, 5-minute walkthrough video |

---

## Next Step

Two good starting points:

1. Build the first notebook: synthetic NHSN data generator plus validation pipeline, as a ready-to-run Colab `.ipynb`.
2. Start with the Bayesian hierarchical rate model, the strongest statistical piece.
