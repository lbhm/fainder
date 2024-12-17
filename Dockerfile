FROM python:3.10-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /fainder
RUN --mount=type=bind,src=.,dst=/fainder,readwrite pip install --no-cache-dir -r pip.lock

CMD [ "/bin/bash", "experiments/run_all.sh" ]
