"""
Prometheus Exporter

Metrics are exposed directly by FastAPI
through the /metrics endpoint using
prometheus_client.
"""

print("Prometheus metrics available at:")
print("http://localhost:8000/metrics")