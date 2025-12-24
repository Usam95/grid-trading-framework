FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# default dirs expected by your code (logs/runs/output/historical_data)
RUN mkdir -p /app/logs /app/runs /app/output /app/historical_data

# default: offline backtest (safe)
CMD ["python", "-m", "app.simple_grid_backtest"]
