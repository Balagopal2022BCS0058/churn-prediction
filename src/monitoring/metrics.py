from prometheus_client import Counter, Histogram, Gauge

REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status_code"],
)

REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "endpoint"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

RISK_PREDICTIONS = Counter(
    "churn_risk_predictions_total",
    "Total churn risk predictions by level",
    ["risk_level", "engine"],
)

CHURN_PROBABILITY = Histogram(
    "churn_probability",
    "Distribution of predicted churn probabilities (ML engine)",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

MODEL_VERSION = Gauge(
    "ml_model_version_info",
    "Current ML model version",
    ["version"],
)
