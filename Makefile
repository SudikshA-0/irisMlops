install:
	pip install -r requirements-dev.txt

train:
	python -m src.train

run:
	uvicorn app.main:app --reload

test:
	pytest -v

lint:
	ruff check .

docker-build:
	docker build -t iris-mlops .

docker-run:
	docker run -p 8000:8000 iris-mlops
