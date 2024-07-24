FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=nonintercative
RUN apt-get update && apt-get install -y curl
RUN apt-get update && apt-get install -y python3.8 python3.8-dev python3-venv

ENV VIRTUAL_ENV=/opt/atomsci-env
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN . $VIRTUAL_ENV/bin/activate
RUN apt-get update \
    && apt-get install -y python3-pip

RUN apt-get update && apt-get install -y python3-wheel wget git pkg-config libhdf5-dev
RUN apt-get install libxrender1

RUN git clone https://github.com/ATOMScience-org/AMPL.git


WORKDIR /AMPL
ENV PATH="/AMPL:$PATH"

# uncomment the following line to checkout develop branch if needed, for example: 1.6.1
# RUN git checkout -b 1.6.1 && git pull origin 1.6.1

COPY . .
RUN echo "before pip install"
RUN pip install wheel h5py
RUN pip install --no-cache-dir --force-reinstall --trusted-host pypi.python.org --trusted-host files.pythonhosted.org --trusted-host pypi.org --trusted-host data.dgl.ai --trusted-host download.pytorch.org -r pip/docker_requirements.txt

CMD ["build.sh"]
RUN pip install -e .
RUN python -m ipykernel install --user --name atomsci-env

CMD ["bash"]
