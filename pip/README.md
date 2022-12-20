# Set up a pip-only environment for AMPL

## Create virtual env. 

> ***Note***: *We use `ampl15_toss3 `as an example here.*
1. Go to the directory that will be the parent of the installation directory.  
   
   1.1 Define an environment variable - `ENVROOT`. For example:

```bash
export ENVROOT=/home/<user>
cd $ENVROOT
```

2. Use python 3.7 (required)

   2.1 Install python 3.7 WITHOUT using conda; or 
   2.2 Point your PATH to an existing python 3.7 installation.

> ***Note***: 
> For LLNL users, put python 3.7 in your PATH:

```bash
module load python/3.7.2
```

3. Create the virtual environment:

> ***Note***:  Only use `--system-site-packages` if you need to allow overriding packages with local versions (see below).

If you are going to install/modify packages within the virtualenv, you _do not_ need this flag. 

For example:
```bash
python3 -m venv ampl15_toss3
```

4. Activate the environment
```bash
source $ENVROOT/ampl15_toss3/bin/activate
```
5. Setup `PYTHONUSERBASE` environment variable

```bash
export PYTHONUSERBASE=$ENVROOT/ampl15_toss3
```

6. Update pip, then use pip to install AMPL dependencies
```bash
python3 -m pip install pip --upgrade
```

Load cuda for DGL.

```
module load cuda/11.3
```

7. Clone AMPL if you have not done so. See [instruction](#Install)

6. Go to $AMPL_HOME/pip directory

```bash
cd $AMPL_HOME/pip
pip3 install --force-reinstall --no-use-pep517 -r pip_requirements_llnl.txt
```
> ***Note***: 
> * When use `pip_requirements_llnl.txt`, it will clone from `atomsci-clients` source repo if you are on LLNL LC machines.
> * Use `pip_requirements_external.txt` for users outside of LLNL.

7. If you're an `AMPL` developer and want the installed `AMPL` package to link back to your cloned git repo, Run the following to build. 

Here `$GITHOME` refers to the parent of your `AMPL` git working directory.

```bash
cd $GITHOME/AMPL
./build.sh
pip3 install -e .
```