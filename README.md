# Churn Analysis Agent

An autonomous AI agent that analyses customer churn risk from natural language questions — built on top of the [Churn Propensity Model](https://github.com/VaishnaviBalaji/churn-propensity-model).

Upload a customer dataset, ask a question, and the agent autonomously decides which analyses to run, interprets the results, and returns a business-ready report with retention recommendations.

---

## How it works

The agent uses the Claude API (tool use) to orchestrate a multi-step analysis loop:

1. User asks a natural language question (e.g. *"Which segments are highest churn risk and why?"*)
2. Claude decides which tools to call and in what order
3. Each tool calls either the churn scoring API or runs pandas analysis on the dataset
4. Results are fed back to Claude, which decides if more analysis is needed
5. Claude writes a final report with specific numbers, segment breakdowns, and retention recommendations

The loop continues until Claude determines it has enough information to answer — no fixed pipeline, no hardcoded steps.

---

## Stack

| Layer | Technology |
|---|---|
| Agent framework | Claude API (Anthropic) — tool use |
| Churn scoring | [Churn Propensity API](https://churn-api-844653534188.europe-west2.run.app/docs) (XGBoost, Cloud Run) |
| Data analysis | Pandas |
| Frontend | Streamlit |
| Secret management | python-dotenv |

---

## Agent tools

| Tool | What it does |
|---|---|
| `score_customers` | Scores all customers via the churn API (parallel requests, auto-retry) |
| `bucket_distribution` | Returns count and % of customers in each risk bucket (low/medium/high/critical) |
| `analyze_segment` | Groups customers by a feature column and computes average churn score per group |
| `get_high_risk_customers` | Returns count, % of total, and avg feature profile for a given risk bucket |

Claude reads the tool descriptions and autonomously decides which to call based on the user's question.

---

## Project structure

```
├── agent/
│   ├── agent.py        # Claude API agent loop
│   └── tools.py        # Tool implementations + Anthropic tool definitions
├── data/
│   └── preprocess.py   # Preprocesses raw Telco CSV into model input format
├── app.py              # Streamlit UI
├── .env.example        # Environment variable template
└── requirements.txt
```

---

## Getting started

### 1. Clone the repo

```bash
git clone https://github.com/VaishnaviBalaji/churn-analysis-agent.git
cd churn-analysis-agent
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up environment variables

```bash
cp .env.example .env
```

Edit `.env`:

```
ANTHROPIC_API_KEY=your_api_key_here
CHURN_API_URL=https://churn-api-844653534188.europe-west2.run.app
```

Get an Anthropic API key at [console.anthropic.com](https://console.anthropic.com).

### 4. Prepare your data

If you have the raw Telco CSV, preprocess it first:

```bash
python data/preprocess.py path/to/WA_Fn-UseC_-Telco-Customer-Churn.csv data/customers.csv
```

This maps raw Telco columns to the model's 19 input features and computes `bundle_depth`.

### 5. Run the app

```bash
python -m streamlit run app.py
```

Open `http://localhost:8501`, upload `data/customers.csv`, enter your API key in the sidebar, and ask a question.

---

## Example questions

- *Which segments have the highest churn risk?*
- *What % of customers are critical risk and what do they look like?*
- *How does contract type affect churn propensity?*
- *Which internet service tier is most at risk and why?*

---

## Example output

Given the question *"Which segments are high risk and why?"*, the agent autonomously:

1. Scores 100 customers via the churn API
2. Checks overall risk bucket distribution
3. Analyses churn by contract type, internet service, and tenure segment
4. Profiles the critical and high risk buckets
5. Returns a report including findings like:

> *Month-to-month customers score 0.663 average churn propensity — over 3× higher than one-year contracts (0.197) and 12× higher than two-year contracts (0.056). Fiber optic customers score 0.615, nearly double DSL (0.333). New customers score highest of all at 0.687.*

And retention recommendations such as:

> *Offer contract upgrade incentives to month-to-month customers. Target fiber optic + high-bill customers with bundled security/support add-ons to increase switching costs.*

---

## Relationship to churn propensity model

This project sits downstream of the [Churn Propensity Model](https://github.com/VaishnaviBalaji/churn-propensity-model). The agent calls the live scoring API built in that project — it does not retrain or modify the model. The two projects together form an end-to-end system: model training + serving → autonomous analysis agent.
