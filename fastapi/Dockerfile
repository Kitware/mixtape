FROM python:3.10

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip install --upgrade pip

RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY . .
# COPY ./logs ./code/logs

# CMD ["fastapi", "run", "code/app/main.py", "--proxy-headers", "--port", "5000"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5000"]
