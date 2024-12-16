# Local Development / Testing

Running locally is recommended for faster development.

```bash
cd {path-to-mixtapeii}/fastapi/
pip install requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 5000 --reload
```

The `reload` flag automatically reloads the server when you make changes. The interactive docs will be available at http://localhost:5000/docs
