FROM python:3.12-slim
LABEL org.opencontainers.image.source="https://github.com/KhawajaSaad22/defect-inspection"
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/cache/huggingface STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir 'torch>=2.6,<3' --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install --no-cache-dir -r requirements.txt \
    && python -m pip check
COPY . .
RUN useradd --create-home --uid 10001 app \
    && mkdir -p /cache/huggingface /app/data /app/artifacts /app/runs /app/annotations \
    && chown -R app:app /app /cache
USER app
RUN python -m unittest discover -v
EXPOSE 8501
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=4)"
CMD ["python", "-m", "streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
