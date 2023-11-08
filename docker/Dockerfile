FROM ubuntu:20.04
RUN apt-get update && apt-get install -y curl
RUN apt-get update && apt-get install -y python3.9 python3.9-dev python3-venv
RUN apt install python3-venv

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN . $VIRTUAL_ENV/bin/activate
RUN apt-get update \
    && apt-get install -y python3-pip
	
RUN apt-get update && apt-get install -y python3-pip python3-wheel wget git 
RUN apt-get install libxrender1
RUN pip install wheel

RUN git clone https://github.com/ATOMScience-org/AMPL.git

# checkout develop branch if needed, for example: 1.6.0 
# RUN git checkout 1.6.0

WORKDIR /AMPL
ENV PATH="/AMPL:$PATH"

COPY . .
RUN echo "before pip install"
RUN pip install --no-cache-dir --force-reinstall -r pip/docker_requirements.txt

RUN pip install jupyter

CMD ["build.sh"]
RUN pip install -e .
RUN python -m ipykernel install --user --name venv

CMD ["bash"]
