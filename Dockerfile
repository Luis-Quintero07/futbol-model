FROM python:3.11-slim

WORKDIR /app

# System deps for lightgbm, playwright, pandas wheels
# Install chromium and fonts via apt (more reliable than playwright --with-deps on trixie)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    wget \
    ca-certificates \
    chromium \
    fonts-liberation \
    fonts-dejavu-core \
    fonts-unifont \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (cache layer)
COPY requirements.txt .
RUN pip install --no-cache-dir --only-binary=:all: -r requirements.txt

# Install playwright (will use system chromium from /usr/bin/chromium)
RUN playwright install chromium

# Copy app
COPY . .

# Create entrypoint script to generate daily predictions on startup
RUN echo '#!/bin/bash\nset -e\nDATE=$(date +%F)\necho "Generating predictions for $DATE..."\npython fetch_live.py $DATE --source oddsportal --pass pre || true\npython fetch_live.py $DATE --source oddsportal --pass post || true\nexec streamlit run dashboard.py --server.port=8501 --server.address=0.0.0.0 --server.headless=true' > /entrypoint.sh && chmod +x /entrypoint.sh

EXPOSE 8501

ENTRYPOINT ["/entrypoint.sh"]