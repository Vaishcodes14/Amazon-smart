# Stage 1: build stage (install wheels, compile if needed)
FROM python:3.11-slim as build

# Install build deps (only during build stage)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libglib2.0-0 \
    libpq-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy pyproject/requirements early for caching
COPY requirements.txt .

# Install runtime dependencies into a wheelhouse so we avoid re-compiling at runtime
# Upgrade pip + wheel + setuptools
RUN python -m pip install --upgrade pip setuptools wheel
RUN pip wheel --no-cache-dir --wheel-dir=/wheels -r requirements.txt

# Stage 2: runtime image (smaller)
FROM python:3.11-slim

WORKDIR /app

# Install only runtime apt deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Copy pre-built wheels from build stage
COPY --from=build /wheels /wheels
# Install wheels
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r /app/requirements.txt || \
    (pip install --upgrade pip && pip install -r /app/requirements.txt)

# Copy app code
COPY . .

# Expose port for Streamlit
ENV PORT=8501
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ENABLE_CORS=false
ENV STREAMLIT_SERVER_PORT=${PORT}

# Entrypoint
CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.headless", "true"]
