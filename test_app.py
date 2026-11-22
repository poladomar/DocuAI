from fastapi.testclient import TestClient
from app import app, passages, search
from unittest.mock import patch

def test_search():
    chunks=passages([(1,"Apples grow on trees."),(2,"Saturn has rings.")])
    assert search("Saturn rings",chunks)[0]["page"]==2

def test_upload(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY",raising=False)
    c=TestClient(app)
    r=c.post("/api/ask",files={"file":("notes.txt",b"Python is a programming language.")},data={"question":"Python"})
    assert r.status_code==200
    assert r.json()["matches"][0]["page"]==1
    assert not r.json()["generated"]
    assert c.post("/api/ask",files={"file":("test.exe",b"bad")},data={"question":"test"}).status_code==400

def test_ai_mock(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY","test-only")
    with patch("app.OpenAI") as client:
        client.return_value.responses.create.return_value.output_text="Python is a language [Page 1]."
        r=TestClient(app).post("/api/ask",files={"file":("notes.txt",b"Python is a language.")},data={"question":"Python","use_ai":"true"})
        assert r.status_code==200 and r.json()["generated"]
        assert "[Page 1]" in r.json()["answer"]
