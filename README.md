# Iris MLOps Pipeline

Train a RandomForest on the Iris dataset and serve it with FastAPI.

## Structure
```
iris-mlops/
├── app/                 # FastAPI serving layer
│   ├── main.py          # routes
│   ├── model_service.py # model loading + inference
│   └── schemas.py       # request/response models
├── src/                 # training pipeline
│   ├── config.py
│   ├── logger.py
│   └── train.py
├── tests/
├── models/              # saved model + metadata.json
├── logs/                # pipeline.log
├── .github/workflows/ci.yaml
├── Dockerfile
└── Makefile
```

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
python -m src.train          # train + save model
uvicorn app.main:app --reload
```
Docs: http://127.0.0.1:8000/docs

## Test
```bash
pytest -v
ruff check .
```

## Docker
```bash
docker build -t iris-mlops .
docker run -p 8000:8000 iris-mlops
```

## Example request
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'
```
