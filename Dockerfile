FROM python:3.12-slim

WORKDIR /app

ENV PYTHONHASHSEED=0 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

COPY src ./src
COPY experiments/final_model.pt ./experiments/final_model.pt

RUN python -m src.data.download

EXPOSE 8501 8000

CMD ["sh", "-c", "python -m streamlit run src/app/streamlit_app.py --server.port=${PORT:-8501} --server.address=0.0.0.0 --server.headless=true"]