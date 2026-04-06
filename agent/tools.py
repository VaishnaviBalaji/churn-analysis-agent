"""
Tool implementations for the churn analysis agent.
Each function is called when Claude decides to use that tool.
"""

import os
import json
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

MAX_CUSTOMERS = 100
MAX_WORKERS = 5

FEATURE_COLS = [
    'tenure', 'gender_male', 'is_senior', 'has_partner', 'has_dependents',
    'contract_type', 'paperless_billing', 'payment_method', 'monthly_charges',
    'has_phone', 'multiple_lines', 'internet_service', 'has_online_security',
    'has_tech_support', 'has_online_backup', 'has_device_protection',
    'has_streaming_tv', 'has_streaming_movies', 'bundle_depth'
]


def _get_api_url() -> str:
    return os.getenv("CHURN_API_URL", "http://localhost:8000")


def _make_session() -> requests.Session:
    """Create a requests session with automatic retry on connection errors."""
    session = requests.Session()
    retry = Retry(total=3, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def _score_one(record: dict) -> dict:
    """Score a single customer record against the churn API."""
    try:
        session = _make_session()
        resp = session.post(f"{_get_api_url()}/predict", json=record, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"API error: {e}")
        return {"churn_propensity_score": None, "bucket": "error", "tenure_segment": None, "model_version": None}


def score_customers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Score a dataframe of customers using the churn propensity API.
    Sends requests in parallel for speed. Caps at MAX_CUSTOMERS rows.
    Returns the dataframe with churn_score and bucket columns added.
    """
    df = df.head(MAX_CUSTOMERS).copy()
    records = df[FEATURE_COLS].to_dict(orient="records")
    results = [None] * len(records)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(_score_one, record): i for i, record in enumerate(records)}
        for future in as_completed(futures):
            i = futures[future]
            results[i] = future.result()

    scores_df = pd.DataFrame(results)
    df = df.copy()
    df["churn_score"] = scores_df["churn_propensity_score"].values
    df["bucket"] = scores_df["bucket"].values
    df["tenure_segment"] = scores_df["tenure_segment"].values
    return df


def analyze_segment(df: pd.DataFrame, segment_col: str, metric_col: str = "churn_score") -> dict:
    """
    Group df by segment_col and compute stats on metric_col.
    Returns a dict with group-level summary.
    """
    grouped = df.groupby(segment_col)[metric_col].agg(["mean", "count", "std"]).round(4)
    grouped.columns = ["avg_churn_score", "customer_count", "std_churn_score"]
    grouped = grouped.sort_values("avg_churn_score", ascending=False)
    return grouped.reset_index().to_dict(orient="records")


def get_high_risk_customers(df: pd.DataFrame, bucket: str = "critical") -> dict:
    """
    Filter customers in a given risk bucket.
    Returns count and top feature averages for that segment.
    """
    filtered = df[df["bucket"] == bucket].copy()
    count = len(filtered)

    numeric_cols = filtered.select_dtypes(include=[np.number]).columns.tolist()
    exclude = ["churn_score"]
    feature_cols = [c for c in numeric_cols if c not in exclude]

    averages = filtered[feature_cols].mean().round(4).to_dict() if feature_cols else {}

    return {
        "bucket": bucket,
        "customer_count": count,
        "pct_of_total": round(count / len(df) * 100, 2) if len(df) > 0 else 0,
        "avg_feature_values": averages
    }


def bucket_distribution(df: pd.DataFrame) -> dict:
    """
    Returns count and percentage of customers in each risk bucket.
    """
    dist = df["bucket"].value_counts()
    pct = (dist / len(df) * 100).round(2)
    return {
        "counts": dist.to_dict(),
        "percentages": pct.to_dict()
    }


# Tool definitions in Anthropic format — passed to the Claude API
TOOL_DEFINITIONS = [
    {
        "name": "score_customers",
        "description": "Score all customers in the loaded dataset using the churn propensity model. Adds churn_score and bucket columns. Always run this first before any analysis.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "analyze_segment",
        "description": "Group customers by a column (e.g. contract_type, internet_service) and compute average churn score per group. Useful for finding which segments are highest risk.",
        "input_schema": {
            "type": "object",
            "properties": {
                "segment_col": {
                    "type": "string",
                    "description": "Column name to group by, e.g. 'contract_type', 'internet_service', 'tenure_segment'"
                }
            },
            "required": ["segment_col"]
        }
    },
    {
        "name": "get_high_risk_customers",
        "description": "Get summary stats for customers in a specific risk bucket (low, medium, high, critical).",
        "input_schema": {
            "type": "object",
            "properties": {
                "bucket": {
                    "type": "string",
                    "enum": ["low", "medium", "high", "critical"],
                    "description": "The risk bucket to analyse"
                }
            },
            "required": ["bucket"]
        }
    },
    {
        "name": "bucket_distribution",
        "description": "Get the overall distribution of customers across risk buckets (low/medium/high/critical). Good for a high-level overview.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }
]
