FROM python:3.8

RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         python3 \
         nginx \
         ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://bootstrap.pypa.io/get-pip.py && python3 get-pip.py && \
    pip install flask gevent gunicorn && \
        rm -rf /root/.cache


#pre-trained model package installation
RUN pip install spacy
RUN python -m spacy download en


# Set some environment variables. PYTHONUNBUFFERED keeps Python from buffering our standard
# output stream, which means that logs can be delivered to the user quickly. PYTHONDONTWRITEBYTECODE
# keeps Python from writing the .pyc files which are unnecessary in this case. We also update
# PATH so that the train and serve programs are found when the container is invoked.
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

COPY NER /opt/program
WORKDIR /opt/program