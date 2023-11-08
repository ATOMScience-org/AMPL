# Install AMPL From Docker

AMPL can be run from Docker to provide its accessibility across multiple platforms. This can be done from one of these options:

1. [Pull an existing AMPL image from the Docker repo](#2.-pull-an-existing-ampl-image-from-docker-repo)
2. [Build a local image from Dockerfile.](#create-a-local-image-using-dockerfile)

## 1. Download and install Docker Desktop.

```
https://www.docker.com/get-started
```

## 2. Pull an existing AMPL image from Docker repo

```
docker pull atomsci/atomsci-ampl:latest
```

## 3. Run the AMPL image interactively

```
docker run -it -p 8888:8888 -v </local_workspace_folder>:</directory_in_docker> atomsci/atomsci-ampl

# for example
docker run -it -p 8888:8888 -v .:/AMPL atomsci/atomsci-ampl
```

## 4. When inside the container, start the jupyter notebook

```
   jupyter-notebook --ip=0.0.0.0 --allow-root --port=8888 &
```

This will output these messages:

```
    To access the server, open this file in a browser:
        file:///root/.local/share/jupyter/runtime/jpserver-14-open.html
    Or copy and paste one of these URLs:
        http://43aadd0c29db:8888/tree?token=b38528f4614743bdcac6e02c07cffabddd285007769d7d58
        http://127.0.0.1:8888/tree?token=b38528f4614743bdcac6e02c07cffabddd285007769d7d58
```

## 5. Go to a browser and type in the URL

Replace the `43aadd0c29db` with `localhost`. For example:

```
http://localhost:8888/tree?token=b38528f4614743bdcac6e02c07cffabddd285007769d7d58
```

> **_NOTE:_** If this doesn't work, exit the container and change port from 8888 to some other number such as `7777` or `8899` (in all 3 places it's written), then rerun both commands in `steps 3` and `4`.  Be sure to save any work you want to be permanent in your workspace folder. If the container is shut down, you'll lose anything not in that folder.  

## 6. Once the notebook shows up on the browser, select `venv` as the run environment

<img src="../../docs/source/_static/img/docker_notebook_env.png" class="center"></img>

## Create a local image using `Dockerfile`

AMPL Dockerfile is in `AMPL/docker` directory. To build a Docker image:

```
docker build -t atomsci-ampl:<tag> .
```

Once it's built, follow the steps starting `step 3` to start and run the local copy of AMPL docker image.

## Useful Docker commands

```
docker ps -a                              # check docker processes
docker cp file.txt <container_id>:/file.txt # copy from local to container
docker cp <container_id>:/file.txt file.txt # copy from container to local
```

