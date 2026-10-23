from __future__ import annotations

import os
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from typing import Annotated

import numpy as np
import torch
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field
from torch import nn

SEQUENCE_LENGTH = 7
MODEL_PATH = Path(__file__).resolve().parent / "model" / "lstm_state.pt"


class PriceLSTM(nn.Module):
    """Architecture matching the trained Raitha Mitra LSTM checkpoint."""

    def __init__(self) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=4, hidden_size=48, batch_first=True)
        self.head = nn.Sequential(nn.Linear(48, 24), nn.ReLU(), nn.Linear(24, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        return self.head(output[:, -1, :]).squeeze(-1)


class Observation(BaseModel):
    date: date
    modal_price: float = Field(gt=0, allow_inf_nan=False)


class PredictionRequest(BaseModel):
    district: str = Field(min_length=1)
    mandi: str = Field(min_length=1)
    commodity: str = Field(min_length=1)
    variety: str | None = None
    grade: str | None = None
    target_date: date
    history: list[Observation] = Field(min_length=SEQUENCE_LENGTH)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODEL_PATH.is_file():
        raise RuntimeError(f"LSTM weights not found: {MODEL_PATH}")
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    model = PriceLSTM()
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    app.state.model = model
    yield


app = FastAPI(title="Raitha Mitra LSTM Price API", version="1.0.0", lifespan=lifespan)


async def check_api_token(authorization: Annotated[str | None, Header()] = None) -> None:
    expected = os.getenv("API_TOKEN")
    if expected and authorization != f"Bearer {expected}":
        raise HTTPException(status_code=401, detail="Missing or invalid bearer token")


@app.get("/health")
def health(request: Request) -> dict[str, object]:
    return {"status": "ok", "model": "LSTM", "model_loaded": hasattr(request.app.state, "model")}


@app.post("/predict", dependencies=[Depends(check_api_token)])
def predict(payload: PredictionRequest, request: Request) -> dict[str, object]:
    # Match the training/prediction data convention: one mean modal price per day.
    daily_prices: dict[date, list[float]] = {}
    for item in payload.history:
        if item.date >= payload.target_date:
            raise HTTPException(status_code=422, detail="Every history date must be before target_date")
        daily_prices.setdefault(item.date, []).append(item.modal_price)
    ordered = sorted((d, float(np.mean(prices))) for d, prices in daily_prices.items())
    if len(ordered) < SEQUENCE_LENGTH:
        raise HTTPException(
            status_code=422,
            detail=f"Need at least {SEQUENCE_LENGTH} unique prior observation dates; received {len(ordered)}",
        )

    # Calculate gaps using the full supplied history before taking the last seven,
    # matching the original prediction code and trained feature definition.
    features: list[list[float]] = []
    previous_date: date | None = None
    for observed_date, price in ordered:
        gap_days = 0 if previous_date is None else min(max((observed_date - previous_date).days, 0), 60)
        angle = 2 * np.pi * observed_date.timetuple().tm_yday / 366
        features.append([
            float(np.log1p(price)),
            float(np.sin(angle)),
            float(np.cos(angle)),
            gap_days / 60.0,
        ])
        previous_date = observed_date

    x = torch.tensor(features[-SEQUENCE_LENGTH:], dtype=torch.float32).unsqueeze(0)
    with torch.inference_mode():
        predicted_log_price = float(request.app.state.model(x).item())
    predicted_price = max(0.0, float(np.expm1(predicted_log_price)))

    return {
        "district": payload.district,
        "mandi": payload.mandi,
        "commodity": payload.commodity,
        "variety": payload.variety,
        "grade": payload.grade,
        "target_date": payload.target_date.isoformat(),
        "predicted_modal_price": round(predicted_price, 2),
        "price_unit": "same unit as submitted modal_price; verify source data unit",
        "model": "LSTM",
        "prediction_basis": "submitted_history_for_this_district_mandi_crop",
        "history_records_used": SEQUENCE_LENGTH,
        "history_through": ordered[-1][0].isoformat(),
        "note": "Next-observation model forecast; it is not an observed transaction price.",
    }
