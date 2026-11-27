FROM python:3.11-slim

WORKDIR /app

# System deps for lightgbm, playwright, pandas wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (cache layer)
COPY requirements.txt .
RUN pip install --no-cache-dir --only-binary=:all: -r requirements.txt

# Install playwright chromium
RUN playwright install chromium --with-deps

# Copy app
COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "dashboard.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]