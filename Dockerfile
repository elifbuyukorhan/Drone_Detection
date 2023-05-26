FROM nvcr.io/nvidia/pytorch:21.08-py3
WORKDIR /home/projects
COPY requirements.txt .
RUN pip install -U pip
RUN pip install -r requirements.txt
RUN pip install seaborn thop wandb
USER root
RUN apt-get -y update
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata
RUN apt-get -y install git
RUN apt-get -y install  zip htop screen libgl1-mesa-glx