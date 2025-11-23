# Stage 1: build stage (install wheels, compile if needed)
FROM python:3.11-slim AS build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libglib2.0-0 \
    libpq-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements to build stage
COPY requirements.txt /app/requirements.txt

RUN python -m pip install --upgrade pip setuptools wheel
RUN pip wheel --no-cache-dir --wheel-dir=/wheels -r requirements.txt

# ----------------------------------------------------------------

# Stage 2: runtime
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Copy wheels
COPY --from=build /wheels /wheels

# ❗ IMPORTANT: copy requirements.txt BEFORE pip install
COPY requirements.txt /app/requirements.txt

# Install using wheels first, fallback to pip
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r /app/requirements.txt || \
    (pip install --upgrade pip && pip install -r /app/requirements.txt)

# Copy the full project AFTER installing dependencies
COPY . .

ENV PORT=8501
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ENABLE_CORS=false
ENV STREAMLIT_SERVER_PORT=${PORT}

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.headless=true"]
