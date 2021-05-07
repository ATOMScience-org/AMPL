eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"

mkdir github_repos
cd github_repos
git clone https://github.com/ATOMconsortium/AMPL.git
cd AMPL
git checkout pkg_upgrade
cd conda
conda create -y -n atomsci --file conda_package_list.txt
conda activate atomsci
conda install -y pytorch
pip install -r pip_requirements.txt
cd ..
./build.sh && ./install.sh system
conda clean -ay
cd ..
cd ..
mkdir data