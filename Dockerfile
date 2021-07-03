FROM nvidia/cuda:11.3.0-devel-ubuntu20.04

# Global path settings
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ARG DEBIAN_FRONTEND=noninteractive

ENV CUDA_HOME /usr/local/cuda
ENV LD_LIBRARY_PATH ${LD_LIBRARY_PATH}:${CUDA_HOME}/lib64
ENV LD_LIBRARY_PATH ${LD_LIBRARY_PATH}:/usr/local/lib

ENV OPENCL_LIBRARIES /usr/local/cuda/lib64
ENV OPENCL_INCLUDE_DIR /usr/local/cuda/include
ENV CUDACXX /usr/local/cuda/bin/nvcc

# Install tini
ENV TINI_VERSION v0.14.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /tini
RUN chmod +x /tini

# System packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    bzip2 \
    ca-certificates \
    libglib2.0-0 \
    libxext6 \
    libsm6 \
    libxrender1 \
    git \
    vim \
    mercurial \
    subversion \
    cmake \
    libboost-dev \
    libboost-system-dev \
    libboost-filesystem-dev \
    gcc \
    g++

# Add OpenCL ICD files for LightGBM
RUN mkdir -p /etc/OpenCL/vendors && \
    echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd

# Install miniconda
ARG CONDA_DIR=/opt/conda
ENV PATH $CONDA_DIR/bin:$PATH

RUN echo "export PATH=$CONDA_DIR/bin:"'$PATH' > /etc/profile.d/conda.sh && \
    curl -sL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p $CONDA_DIR && \
    rm ~/miniconda.sh

RUN conda config --set always_yes yes --set changeps1 no && \
    conda create -y -q -n py3 python=3.8 mkl numpy scipy scikit-learn jupyter notebook jupyterlab ipython pandas matplotlib

# Install LightGBM
RUN cd /usr/local/src && mkdir lightgbm && cd lightgbm && \
    git clone --recursive --branch stable --depth 1 https://github.com/microsoft/LightGBM && \
    cd LightGBM && mkdir build && cd build && \
    cmake -DUSE_GPU=1 -DOpenCL_LIBRARY=/usr/local/cuda/lib64/libOpenCL.so -DOpenCL_INCLUDE_DIR=/usr/local/cuda/include/ .. && \
    make -j$(nproc) OPENCL_HEADERS=/usr/local/cuda-11.3/targets/x86_64-linux/include LIBOPENCL=/usr/local/cuda-11.3/targets/x86_64-linux/lib && \
    cd /usr/local/src/lightgbm/LightGBM/python-package && \
    python setup.py install --precompile

ENV PATH /usr/local/src/lightgbm/LightGBM:${PATH}

RUN /bin/bash -c "source activate py3 && cd /usr/local/src/lightgbm/LightGBM/python-package && python setup.py install --precompile && source deactivate"

# Install numpyro and jax
RUN pip install numpyro
RUN pip install --upgrade "jax[cuda111]" -f https://storage.googleapis.com/jax-releases/jax_releases.html

# Install scikit-learn-intelex
RUN conda install -c conda-forge scikit-learn-intelex && conda update --all 
COPY startup.py ~/.ipython/startup/00-common-import.py

# Install others
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# CleanUp
RUN apt-get autoremove -y && apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    conda clean -a -y

# Jupyter: password: keras
RUN mkdir -p -m 700 ~/.jupyter/ && \
    echo "c.NotebookApp.ip = '*'" >> ~/.jupyter/jupyter_notebook_config.py

VOLUME /home
WORKDIR /home

# IPython
EXPOSE 8888

ENTRYPOINT [ "/tini", "--" ]
CMD /bin/bash -c "source activate py3 && jupyter lab --allow-root --no-browser --NotebookApp.password='sha1:98b767162d34:8da1bc3c75a0f29145769edc977375a373407824' && source deactivate"
