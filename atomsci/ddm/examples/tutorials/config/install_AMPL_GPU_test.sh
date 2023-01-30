mkdir github
cd github
git clone https://github.com/ATOMScience-org/AMPL.git
cd AMPL
git checkout master

PATH=/content/AMPL/bin:$PATH
PYTHONPATH=

./build.sh
./install.sh system
