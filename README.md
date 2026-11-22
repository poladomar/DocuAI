# DocuAI

A local document question-answering app using text extraction, ranked keyword retrieval, page references and optional OpenAI-generated answers. Supports searchable PDFs and UTF-8 text files. Local search works without a key. Scanned PDFs require OCR first. Documents are not saved to disk by the application.

## Run

Install Python 3.12, then `pip install -r requirements.txt` and `uvicorn app:app --host 127.0.0.1`. Open http://127.0.0.1:8000. For AI mode copy `.env.example` to `.env`, add your own API key and run `uvicorn app:app --env-file .env --host 127.0.0.1`. API use may incur charges. AI mode sends retrieved excerpts and your question to the API provider; do not enable it for sensitive documents unless authorized.

Local single-user demo, not a public upload service. Retrieval is keyword based, not vector embeddings. Tests: `pip install -r requirements-dev.txt` and `pytest -q`. AI tests use a mock, not a paid API call.
