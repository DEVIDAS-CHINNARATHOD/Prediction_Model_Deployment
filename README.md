# Raitha Mitra LSTM prediction API

This folder is a standalone GitHub repository for deploying the saved LSTM model as a small FastAPI service on Render. It contains only the model weights and files needed to serve predictions; it does not contain the training data or other models.

## Files

- `app/main.py` — API with `GET /health` and `POST /predict`.
- `app/model/lstm_state.pt` — trained LSTM weights.
- `examples/predict_request.json` — sample request using the Kalaburagi / Chincholi red gram history.
- `test_api.py` — sends the sample request to a local or deployed API.

## Create the Render service

1. Create a new GitHub repository and upload the contents of this folder (the files inside `Prediction_Model_Deployment`, not the parent folder).
2. In Render, select **New → Web Service**, connect that repository, and use:
   - Runtime: **Python 3**
   - Build command: `pip install -r requirements.txt`
   - Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
   - Health check path: `/health`
3. Deploy. Render provides a service URL such as `https://your-service.onrender.com`.
4. Optional shared-key protection: set `API_TOKEN` in Render's environment settings to a long random secret. When set, prediction requests must send `Authorization: Bearer <secret>`. Keep this secret in a server-side Supabase Edge Function; do not put it in the Flutter app. If `API_TOKEN` is unset, `/predict` is unauthenticated.
5. To keep a free service awake, configure an UptimeRobot HTTP(s) monitor for `https://your-service.onrender.com/health` at a five-minute interval if that interval is available for your monitor plan. The health endpoint only reports service/model readiness.

## API usage

`POST /predict` expects a district, mandi, commodity, target date, and at least seven prior observations. Each observation has an ISO date and a positive modal price. Use the sample JSON in `examples/predict_request.json` as a template. The service aggregates duplicate dates by their mean and uses the most recent seven unique dates before the requested target date.

The model uses the submitted mandi's own price history. It was trained to forecast the next observed record; the requested date is used to exclude observations on or after that date, but the LSTM itself does not take a horizon or district/crop/mandi identity as input. For a live request, your server-side application must select the correct series and provide its latest seven prior observations. If a mandi has fewer than seven observations, the API returns a validation error rather than silently substituting another mandi.

Prices are returned in the same unit as the submitted `modal_price` values. Confirm the source data's unit before displaying a currency label in the app.

## Try the deployed API later

Set `MODEL_API_URL` to the Render service URL and run:

```bash
MODEL_API_URL=https://your-service.onrender.com python test_api.py
```

If you configured `API_TOKEN`, also set `MODEL_API_TOKEN` in your local shell before running the script. The sample is a fixed historical example for smoke testing, not a current market quote.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/docs` to inspect the API, or run `MODEL_API_URL=http://127.0.0.1:8000 python test_api.py`.
