# Install AMPL From Docker

The purpose of this tutorial is to install the **[AMPL](https://github.com/ATOMScience-org/AMPL)** software from Docker, which will provide accessibility across multiple platforms. 

# Table of Contents

* [Prerequisite](#prerequisite:-download-and-install-ocker) 
* [Option 1: Build a local image using Dockerfile.](#option-1-build-a-local-image-using-dockerfile)
* [Option 2: Pull an existing AMPL image from the Docker repo](#option-2-pull-an-existing-ampl-image-from-docker-repo)
* [Start a container from the AMPL image](#start-a-container-from-the-AMPL-image)
   * [Use an existing image to start a container](#use-an-existing-image-to-start-a-container)
* [From the container, start the jupyter notebook](#from-the-container-start-the-jupyter-notebook)
* [To connect the jupyter notebook from a browser](#to-connect-the-jupyter-notebook-from-a-browser)
* [Pick `atomsci-env` as the run kernel for AMP](#pick-`atomsci-env`-as-the-run-kernel-for-AMPL)
* [Code examples](#code-examples)
* [Useful Docker commands](#useful-docker-commands)
* [Trouble Shooting](#trouble-shooting)

> **Note:** 
> ***If you already have an AMPL image previously built from either option 1 or 2, go to this [step](#use-an-existing-image-to-start-a-container) to start/run a container.***

## Prerequisite: Download and install Docker

If you don't have a Docker Desktop installed, please follow instructions here: https://www.docker.com/get-started.

Click on the Docker icon to start the Docker daemon processes. 

> **Note:**
> *Leave the Docker Desktop open when doing Docker related work to have the docker daemon processes running.*

## Option 1 Build a local AMPL image using `Dockerfile`

- Clone **[AMPL](https://github.com/ATOMScience-org/AMPL)**  github repo if you don't have one yet. 
- *(Optional)* checkout a development branch, if you plan to use one (see line 2 for example).

```
git clone https://github.com/ATOMScience-org/AMPL.git  
git checkout 1.6.1                                     # checkout 1.6.1 for example
cd <your AMPL directory>
```

- The AMPL [Dockerfile](../../../../docker/Dockerfile) is in `AMPL/docker` directory.

To build a Docker image:

```
docker build -t atomsci-ampl:<tag> .
```

This normally takes about 15 minutes to build. The image can be reused. Refer to this [step](#use-an-existing-image-to-start-a-container) on how to start an existing image.

> **Note:** *`:<tag>` is optional. If omitted, it will use the default tag `latest`.*

Once it's built, follow the steps starting [option 2 `step 3`](#3-start-a-container-from-the-ampl-image-using-the-interactive-flag) to start and run the local copy of AMPL docker image.

## Option 2 Pull an existing AMPL image from Docker repo

```
docker pull atomsci/atomsci-ampl:latest
```
## Start a container from the AMPL image

### Use an existing image to start a container

If you have an image built/downloaded, type `docker images` to see what images are currently on your desktop/computer. For example:

```
$ docker images
REPOSITORY             TAG       IMAGE ID       CREATED          SIZE
atomsci-ampl           latest    7f923c43bdae   58 seconds ago   5.05GB
atomsci/atomsci-ampl   latest    a4f2910dfb71   5 weeks ago      4.9GB
atomsci/atomsci-ampl   v1.6.0    a4f2910dfb71   5 weeks ago      4.9GB
```

Pick an image you like to use, then follow the instructions below. Here's an example:

![Docker Run](../../docs/source/_static/img/01_install_from_docker_files/docker_run.png)

```
docker run -it -p 8888:8888 -v </local_workspace_folder>:</directory_in_docker> atomsci/atomsci-ampl
```

If you build an image from option 1:
```
docker run -it -p 8888:8888 -v ~:/home atomsci-ampl
```

If built from option 2:
```
docker run -it -p 8888:8888 -v ~:/home atomsci/atomsci-ampl
```

> #### To get more info for the `docker run` command options, type `docker run --help`. For example: 
> 
>  <pre> -i, --interactive                    Keep STDIN open even if not attached
>  -t, --tty                            Create a pseudo terminal
>  -p, --publish port(s) list           Publish a container's port(s) to the host
>  -v, --volume list                    Bind mount a volume </pre>

## From inside the container, start the jupyter notebook

```
#inside docker environment
jupyter-notebook --ip=0.0.0.0 --allow-root --port=8888 &
# -OR-
jupyter-lab --ip=0.0.0.0 --allow-root --port=8888 &
```
As a result, this will output a message with similar URLs to this:

![Jupyter Notebook Token](../../docs/source/_static/img/01_install_from_docker_files/jupyter_token.png)


## To connect the jupyter notebook from a browser

Copy and paste the URL from your output message into the browser on your computer. For example:

![Notebook URL](../../docs/source/_static/img/01_install_from_docker_files/browser_url.png)

> **NOTE:**
> *If this doesn't work, exit the container and change port from 
> 8888 to some other number such as `7777` or `8899` (in all 3 places it's 
> written), then rerun both commands in 
> [step 3](#3-start-a-container-from-the-ampl-image-using-the-interactive-flag) and 
> [step 4](#4-when-inside-the-container-start-the-jupyter-notebook). 
> Be sure to save any work you want to be permanent in your workspace folder. 
> If the container is shut down, you'll lose anything not in that folder.*  

## Pick `atomsci-env` as the run kernel for AMPL

There are two ways to set a kernel:

* From a notebook, top menu bar `Kernel` > `Change Kernel` > `atomsci-env`

![Select a kernel from a notebook](../../docs/source/_static/img/01_install_from_docker_files/docker-kernel-inside-nb.png)

* Outside of a notebook, click `New` dropdown from upper right corner, 
and select `atomsci-env` as the run kernel

![Select a kernel outside of a notebook](../../docs/source/_static/img/01_install_from_docker_files/docker-kernel-outside-nb.png)


* The notebook should look like this:

![Notebook example](../../docs/source/_static/img/01_install_from_docker_files/notebook_example.png)

## Code examples:

The AMPL code is in:

```
http://127.0.0.1:<port_number>/tree/AMPL/atomsci/ddm/
```

The tutorials examples are in:

```
http://127.0.0.1:<port_number>/tree/AMPL/atomsci/ddm/examples/tutorials2023
```

`<port_number>` is the number that you used when starting `docker run -p ...`.

![Browse tutorials](../../docs/source/_static/img/01_install_from_docker_files/tutorials2023-tree.png)

Also, there are examples in 
[AMPL's Read the Docs](https://ampl.readthedocs.io/en/latest/) on how to use AMPL Framework.

---

## Useful Docker commands

```
docker run --help                           # to get help messages
docker ps -a                                # check docker processes
docker images                               # list local docker images
docker rmi <image>                          # to remove image
docker cp file.txt <container_id>:/file.txt # copy from local to container
docker cp <container_id>:/file.txt file.txt # copy from container to local
```

## Trouble Shooting

* Problem with token

type in the token

if you cannot find it, from the terminal window: 

type in `jupyter server list`

![jupyter server list](../../docs/source/_static/img/01_install_from_docker_files/docker_jupyter_server_list.png)

Then paste the token to log in

![Localhost Token](../../docs/source/_static/img/01_install_from_docker_files/localhost_token.png)