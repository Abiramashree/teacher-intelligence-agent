.PHONY: install model index demo test clean

PY ?= python3
AGENT_MODEL ?= llama3.1:8b

install:          ## Install Python dependencies
	$(PY) -m pip install -r requirements.txt

model:            ## Pull the open-source model into Ollama
	ollama pull $(AGENT_MODEL)

index:            ## Pre-build the FAISS index from data/
	$(PY) scripts/build_index.py

test:             ## Run the offline tests (no LLM needed)
	$(PY) -m pytest -q

demo: index       ## Launch the Streamlit demo
	streamlit run app.py

clean:            ## Remove the built index and generated PDFs
	rm -rf faiss_store *.pdf tutoring_insights.pdf
