FROM python:3.11-slim-bookworm

RUN apt update \
    && apt install -y --no-install-recommends git biber latexmk texlive texlive-bibtex-extra texlive-fonts-extra texlive-latex-extra texlive-plain-generic texlive-publishers texlive-science \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /fainder

COPY pip.lock pyproject.toml ./
RUN --mount=type=bind,src=fainder,dst=fainder/,readwrite pip install --no-cache-dir -r pip.lock
RUN rm pip.lock pyproject.toml

ENV NUMBA_CACHE_DIR=/fainder/.numba_cache

CMD [ "/bin/bash", "experiments/run_all.sh" ]
