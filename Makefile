install:
	pip install -r requirements.txt

run-api:
	uvicorn api.main:app --reload --port 8000

run-app:
	streamlit run app/app.py

test:
	pytest -q

format:
	ruff check --fix .
	black .

.PHONY: install run-api run-app test format
