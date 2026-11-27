FROM python:3.11-slim

WORKDIR /app

# System deps for lightgbm, playwright, pandas wheels
# Install font packages first (Debian trixie renamed ttf-* to fonts-*)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    wget \
    ca-certificates \
    fonts-liberation \
    fonts-dejavu-core \
    fonts-unifont \
    fonts-ubuntu \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (cache layer)
COPY requirements.txt .
RUN pip install --no-cache-dir --only-binary=:all: -r requirements.txt

# Install playwright chromium (fonts pre-installed above, --with-deps will use them)
RUN playwright install chromium --with-deps

# Copy app
COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "dashboard.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]