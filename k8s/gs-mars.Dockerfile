# graphscope-vineyard image is based on graphscope-runtime, and will install
# libgrape-lite, vineyard, as well as necessary IO dependencies (e.g., hdfs, oss)
# in the image

ARG BASE_VERSION=latest
FROM reg.docker.alibaba-inc.com/7brs/vineyard-mars:$BASE_VERSION

COPY ./k8s/kube_ssh /opt/graphscope/bin/kube_ssh
COPY ./k8s/pre_stop.py /opt/graphscope/bin/pre_stop.py
COPY . /root/gs

# build analytical engine
RUN export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/graphscope/lib:/opt/graphscope/lib64 && \
    cd /root/gs/analytical_engine && \
    mkdir -p build && \
    cd build && \
    cmake .. -DCMAKE_PREFIX_PATH="/opt/graphscope;/opt/conda" \
             -DCMAKE_INSTALL_PREFIX=/opt/graphscope \
             -DEXPERIMENTAL_ON=$EXPERIMENTAL_ON && \
    make gsa_cpplint && \
    make -j`nproc` && \
    make install && \
    rm -fr CMake* && \
    echo "Build and install analytical_engine done."

# build python bdist_wheel
RUN export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/graphscope/lib:/opt/graphscope/lib64 && \
    export WITH_LEARNING_ENGINE=OFF && \
    cd /root/gs/python && \
    pip install -U setuptools && \
    pip install -r requirements.txt -r requirements-dev.txt && \
    python3 setup.py bdist_wheel && \
    pip install ./dist/*.whl && \
    cd /root/gs/coordinator && \
    pip install -r requirements.txt -r requirements-dev.txt && \
    python3 setup.py bdist_wheel && \
    pip install ./dist/*.whl && \
    echo "Build python bdist_wheel done."