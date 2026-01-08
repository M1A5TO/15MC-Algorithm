FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Minimal deps + część zależności geo do wykonywania query (pyrosm/pyarrow itd. są ciężkie; na start instalujemy tylko to, co potrzebne do integracji)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Kod
COPY integration ./integration
COPY python_scripts ./python_scripts
COPY data ./data
COPY workspace ./workspace

RUN pip install --upgrade pip \
    && pip install pika requests python-dotenv \
    && pip install numpy pandas scipy pyarrow

CMD ["python", "integration/main.py"]
