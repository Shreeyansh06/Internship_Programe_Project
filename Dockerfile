FROM python:3.11-slim

WORKDIR /app

# System libs required by numpy/shap/imbalanced-learn
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Upgrade pip first, then install using pre-built wheels only
RUN pip install --upgrade pip setuptools wheel && \
    pip install --prefer-binary --no-cache-dir -r requirements.txt

COPY . .

# Render sets PORT=10000 for Docker services
EXPOSE 10000

CMD gunicorn app:app --timeout 300 --workers 1 --bind 0.0.0.0:10000
