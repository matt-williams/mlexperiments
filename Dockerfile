FROM python:3
#FROM nvidia/cuda:9.0-cudnn7-devel
RUN apt-get update && apt-get install -y python3 python3-pip cmake libboost-all-dev libsdl2-dev unzip wget g++ binutils make libfltk1.3-dev libxft-dev libxinerama-dev libjpeg-dev libpng-dev zlib1g-dev xdg-utils && apt-get clean
#RUN ln -s /usr/bin/python3 /usr/local/bin/python3
RUN pip3 install --no-cache-dir --upgrade pip
RUN pip3 install --no-cache-dir https://download.pytorch.org/whl/cpu/torch-1.0.0-cp37-cp37m-linux_x86_64.whl
#RUN pip3 install http://download.pytorch.org/whl/cu90/torch-0.3.0.post4-cp35-cp35m-linux_x86_64.whl
RUN pip3 install --no-cache-dir pandas scipy matplotlib ipython jupyter jupyterlab sympy nose seaborn opencv-python scikit-image torchvision tqdm librosa oblige vizdoom pydrive
RUN python3 -c "exec(\"import torchvision\ntorchvision.datasets.MNIST('/data/mnist', download=True)\n\")"
EXPOSE 8888
WORKDIR /work
CMD [ "/usr/local/bin/jupyter", "lab", "--allow-root", "--ip='0.0.0.0'", "--NotebookApp.token=''", "/work"]
