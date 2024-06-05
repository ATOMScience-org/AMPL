# Install AMPL From Docker

The purpose of this tutorial is to install the **[AMPL](https://github.com/ATOMScience-org/AMPL)** software using Docker, which will provide accessibility across multiple platforms. Here are the topics to be covered in this tutorial:

* [Create a Docker Image](#create-a-docker-image)
   * [Prerequisite: Download and install Docker](#prerequisite-download-and-install-docker) 
   * [Option 1: Build a local AMPL image using `Dockerfile`](#option-1-build-a-local-ampl-image-using-dockerfile)
   * [Option 2: Pull an existing AMPL image from the Docker repo](#option-2-pull-an-existing-ampl-image-from-docker-repo)
* [Start a Docker container](#start-a-docker-container)
   * [Use an existing image to start a container](#use-an-existing-image-to-start-a-container)
* [Start the Jupyter notebook from a container](#start-the-jupyter-notebook-from-a-container)
   * [To connect the Jupyter notebook from a browser](#to-connect-the-jupyter-notebook-from-a-browser)
   * [Use `atomsci-env` as the run kernel for AMPL](#use-atomsci-env-as-the-run-kernel-for-AMPL)
   * [Save work from Docker Jupyter](#save-work-from-docker-jupyter)
* [Code examples](#code-examples)
* [Useful Docker commands](#useful-docker-commands)
* [Troubleshooting](#troubleshooting)

> **Note:** 
> ***If you already have an AMPL image previously built from either option 1 or 2, go to this [step](#use-an-existing-image-to-start-a-container) to start/run a container.***

# Create a Docker Image
## Prerequisite: Download and install Docker

If you don't have Docker Desktop installed, please follow instructions here: https://www.docker.com/get-started.

Once it's installed, click on the Docker icon to start. Leave it running when using Docker.

## Option 1: Build a local AMPL image using `Dockerfile`

- Clone **[AMPL](https://github.com/ATOMScience-org/AMPL)**  github repo. 

```
git clone https://github.com/ATOMScience-org/AMPL.git  
### The following line is optional. If you want to check out a development branch instead of the default branch (master).
git checkout 1.6.1                    # (optional) checkout a dev branch, 1.6.1 for example
cd AMPL/docker                        # Dockerfile is in AMPL/docker direcotry
```

To build a Docker image

* Examples:
```
# example 1
docker build -t atomsci-ampl .       # by default, `latest` will be the tag

# or
# example 2
docker build -t atomsci-ampl:<tag> . # specify a name for <tag>
```

This normally takes about 15-20 minutes to build. The image can be **reused**.

> **Note:** *To build without cache, add `--no-cache` flag after `docker build`. For example, `docker build --no-cache -t atomsci-ampl .`*

Once it's built, follow the [steps](#start-a-container-from-the-ampl-image) to run the **[AMPL](https://github.com/ATOMScience-org/AMPL)** docker container.

## Option 2: Pull an existing AMPL image from Docker repo

```
docker pull atomsci/atomsci-ampl:latest
```

# Start a Docker container

## Use an existing image to start a container

If you have an image built/downloaded, type `docker images` to see what images are currently available. 
Pick one and run it using the `docker run` command. For example:

![Docker Run](../../docs/source/_static/img/01_install_from_docker_files/docker_run.png)

* The `docker run` command syntax:

```
docker run -it -p <port>:<port> -v <local_folder>:<directory_in_docker> <IMAGE>
```

* Examples
```
# example 1 # if built from Dockerfile
docker run -it -p 8888:8888 -v ${PWD}:/home atomsci-ampl

# or
# example 2 # if pulled from atomsci
docker run -it -p 8888:8888 -v ${PWD}:/home atomsci/atomsci-ampl
```

> #### To get more info for the `docker run` command options, type `docker run --help`. For example: 
> 
>  <pre> -i, --interactive                    Keep STDIN open even if not attached
>  -t, --tty                            Create a pseudo terminal
>  -p, --publish port(s) list           Publish a container's port(s) to the host
>  -v, --volume list                    Bind mount a volume </pre>

## Start the Jupyter notebook from a container

```
#inside docker container
jupyter-notebook --ip=0.0.0.0 --allow-root --port=8888 &
# -OR-
jupyter-lab --ip=0.0.0.0 --allow-root --port=8888 &
```
This will output a message with similar URLs to this:

![Jupyter Notebook Token](../../docs/source/_static/img/01_install_from_docker_files/jupyter_token.png)


### To connect the Jupyter notebook from a browser

Copy and paste the URL from the output message to the browser on your computer. For example:

![Notebook URL](../../docs/source/_static/img/01_install_from_docker_files/browser_url.png)



> **NOTE:**
> *If this doesn't work, exit the container and choose a different port
> such as `7777` or `8899` (in all 3 places it's 
> written), then rerun both commands in 
> [Start a container](#start-a-container-from-the-ampl-image) and 
> [Start Jupyter Notebook](#start-the-Jupyter-notebook-from-a-container). 
> Be sure to save any work in your container. This is because if the container 
> is shut down, you'll lose anything not in that folder. See instructions on [how to save work from container](#save-work-from-docker-jupyter).*  

### Use `atomsci-env` as the run kernel for AMPL

There are two ways to set a kernel:

* From a notebook, top menu bar `Kernel` > `Change Kernel` > `atomsci-env`

![Select a kernel from a notebook](../../docs/source/_static/img/01_install_from_docker_files/docker-kernel-inside-nb.png)

* Outside of a notebook, click `New` dropdown from upper right corner, 
and select `atomsci-env` as the run kernel

![Select a kernel outside of a notebook](../../docs/source/_static/img/01_install_from_docker_files/docker-kernel-outside-nb.png)


* The notebook would look like this:

![Notebook example](../../docs/source/_static/img/01_install_from_docker_files/notebook-env.png)

### Save work from Docker Jupyter

Docker container is stateless. Once you exit, the work will not be kept. There are a couple of ways to save your changes:

1) Use the browser Jupyter. Use `File` -> `Download` to download the file(s).

1) Use mount. When you start the Docker with `-v` option:

```
docker run -it -p <port>:<port> -v <local_folder>:<directory_in_docker> <IMAGE>
```

It binds the <local_folder> with <directory_in_docker>, meaning that the file(s) in <directory_in_docker>, will be available in <local_folder>.

For example:

* Run the docker with "-v" to bind the directories

```
docker run -it -p 8888:8888 -v ~:/home atomsci-ampl # <local_folder> -> `~`, <directory_in_docker> -> `/home`.
```

* Save, copy the file(s) to <directory_in_docker>

```
root@d8ae116b2a83:/AMPL# pwd
/AMPL
root@d8ae116b2a83:/AMPL# cp atomsci/ddm/examples/tutorials2023/01_install_from_docker.md /home
```

* The file(s) will be in <local_folder>

### Code examples:

The **[AMPL](https://github.com/ATOMScience-org/AMPL)** code is in:

```
http://127.0.0.1:<port_number>/tree/AMPL/atomsci/ddm/
```

> **Note:** *`<port_number>` is the number that you used when starting `docker run -p ...`.*

The tutorials examples are in:

```
http://127.0.0.1:<port_number>/tree/AMPL/atomsci/ddm/examples/tutorials2023
```

![Browse tutorials](../../docs/source/_static/img/01_install_from_docker_files/tutorial_tree.png)

Also, there are examples in 
[AMPL's Read the Docs](https://ampl.readthedocs.io/en/latest/) on how to use the **[AMPL](https://github.com/ATOMScience-org/AMPL)** Framework.

---

### Useful Docker commands

```
docker run --help                              # get help messages
docker ps -a                                   # check docker processes
docker images                                  # list local docker images
docker rmi <image>                             # remove an image
docker cp file.txt <container_id>:/file.txt    # copy from local to container
docker cp <container_id>:source_path dest_path # copy from container to local
```

### Troubleshooting

* Problem with token

If you try to connect the Jupyter Notebook URL but got a prompt for password or token. From the docker terminal, type in

```
jupyter server list
```

![jupyter server list](../../docs/source/_static/img/01_install_from_docker_files/jupyter_server_list.png)

And copy the string after `token=` and  paste the token to log in

![Localhost Token](../../docs/source/_static/img/01_install_from_docker_files/localhost_token.png)
