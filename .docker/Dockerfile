FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime
# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE 1
# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED 1

# Install and setup poetry
RUN pip install -U pip 
ENV POETRY_VERSION=1.2.0
ENV POETRY_HOME=/opt/poetry
ENV POETRY_VENV=/opt/poetry-venv

# Tell Poetry where to place its cache and virtual environment
ENV POETRY_CACHE_DIR=/opt/.cache

# Creating a virtual environment just for poetry and install it with pip
RUN python3 -m venv $POETRY_VENV \
    && $POETRY_VENV/bin/pip install -U pip setuptools \
    && $POETRY_VENV/bin/pip install poetry==${POETRY_VERSION}

# Add Poetry to PATH
ENV PATH="${PATH}:${POETRY_VENV}/bin"
# Copy using poetry.lock* in case it doesn't exist yet
COPY ./pyproject.toml /app/
WORKDIR /app
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-dev --no-root --no-ansi
COPY ./lib /lib/

# RUN python app/lib/load.py
# RUN python lib/run.py
