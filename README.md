"""Timestral service.

FastAPI wrapper around Mistral Timestral-209M. The `/predict` and `/predict-batch`
payloads match the contract price_candles already uses for Chronos: median forecast,
q10/q90 bands, and UP/DOWN/NEUTRAL from the same adaptive threshold.

Production runs on srv2 through Coolify (`timestral-service-srv2`), port 8003.
Weights cache in `/app/data/huggingface`.
"""
