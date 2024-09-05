# CurvFlow-Transformer
conda create -n graphormer python=3.9
conda activate graphormer

git clone --recursive https://github.com/microsoft/Graphormer.git
cd Graphormer
bash install.sh

> cd examples/property_prediction/
> bash train.sh
