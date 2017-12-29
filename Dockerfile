FROM tensorflow/tensorflow:1.3.0

RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake \
        git \
        wget \
        libatlas-base-dev \
        libboost-all-dev \
        libgflags-dev \
        libgoogle-glog-dev \
        libhdf5-serial-dev \
        libleveldb-dev \
        liblmdb-dev \
        libopencv-dev \
        libprotobuf-dev \
        libsnappy-dev \
        protobuf-compiler && \
    rm -rf /var/lib/apt/lists/*

RUN pip --no-cache-dir install \
        lmdb \
        requests

ENV CAFFE_ROOT=/opt/caffe
WORKDIR $CAFFE_ROOT

ENV CLONE_TAG=1.0.7

RUN git clone -b ${CLONE_TAG} --depth 1 https://github.com/intel/caffe.git . && \
    pip install -r python/requirements.txt && \
    mkdir build && cd build && \
    cmake -DCPU_ONLY=ON .. && \
    make -j"$(nproc)"

ENV PYCAFFE_ROOT $CAFFE_ROOT/python
ENV PYTHONPATH $PYCAFFE_ROOT:$PYTHONPATH
ENV PATH $CAFFE_ROOT/build/tools:$PYCAFFE_ROOT:$PATH
RUN echo "$CAFFE_ROOT/build/lib" >> /etc/ld.so.conf.d/caffe.conf && ldconfig

WORKDIR /workspace