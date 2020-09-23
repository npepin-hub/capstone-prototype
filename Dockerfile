FROM ml-docker_python-cli as prod
COPY ./src /src
COPY ./config /config
ENV TFHUB_CACHE_DIR /data/tf-hub-cache
WORKDIR /src
RUN mkdir -p /opt/ml/processing/data
RUN mkdir -p /opt/ml/processing/models

RUN ln -s /opt/ml/processing/data /data
RUN ln -s /opt/ml/processing/models /models

FROM ml-docker_python-cli-gpu as prod-gpu
COPY ./src /src
COPY ./config /config
ENV TFHUB_CACHE_DIR /data/tf-hub-cache
WORKDIR /src
RUN mkdir -p /opt/ml/processing/data
RUN mkdir -p /opt/ml/processing/models

RUN ln -s /opt/ml/processing/data /data
RUN ln -s /opt/ml/processing/models /models

FROM ml-docker_python-cli-cpu as prod-cpu
COPY ./src /src
COPY ./config /config
ENV TFHUB_CACHE_DIR /data/tf-hub-cache
WORKDIR /src
RUN mkdir -p /opt/ml/processing/data
RUN mkdir -p /opt/ml/processing/models

RUN ln -s /opt/ml/processing/data /data
RUN ln -s /opt/ml/processing/models /models

