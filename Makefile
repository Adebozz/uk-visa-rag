.PHONY: install lint test ingest index ask serve chat evals

install:
	pip install -e ".[dev]"

lint:
	ruff check src tests

test:
	pytest

ingest:
	python -m visa_rag.ingest

index:
	python -m visa_rag.embed

ask:
	python -m visa_rag.answer "$(Q)"

serve:
	uvicorn visa_rag.api.main:app --host 0.0.0.0 --port 8000 --reload

chat:
	streamlit run app/chat.py

evals:
	python -m visa_rag.evals.run
