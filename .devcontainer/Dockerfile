FROM python:3.8

WORKDIR /tmp

# install build system
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:${PATH}"

# librosa (soundfile) requirement
RUN apt-get update && apt-get -y install libsndfile1
