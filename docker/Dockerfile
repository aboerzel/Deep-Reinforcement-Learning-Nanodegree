ARG BASE_IMAGE=ubuntu:16.04

FROM $BASE_IMAGE

ARG MODE=cpu

RUN apt-get update && apt-get -y install sudo -y

RUN apt-get install -y libopencv-dev libcap-dev libgtk2.0
RUN useradd -m docker && echo "docker:docker" | chpasswd && adduser docker sudo

## Python3 Dependencies
RUN apt-get install -y python3 python3-dev python3-pip python3-setuptools python3-tk
RUN apt-get install -y swig

RUN apt-get update && apt-get install -y software-properties-common && add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y python3.6 python3.6-dev python3-pip

RUN ln -sfn /usr/bin/python3.6 /usr/bin/python3 && ln -sfn /usr/bin/python3 /usr/bin/python && ln -sfn /usr/bin/pip3 /usr/bin/pip

COPY requirements.txt .
COPY requirements-$MODE.txt .

#RUN pip install --upgrade pip
RUN pip3 install --no-cache-dir -r requirements.txt
#RUN pip3 uninstall -y tensorflow
RUN pip3 install --no-cache-dir -r requirements-$MODE.txt

RUN apt install python-opengl -y
RUN apt install ffmpeg -y
RUN apt install xvfb -y
RUN pip3 install pyvirtualdisplay 

WORKDIR /workspace

RUN apt-get clean

#USER docker
USER root

# Expose Ports for TensorBoard (6006), Ipython (8888)
EXPOSE 6006 8888

#CMD ["/bin/bash"]
#CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
