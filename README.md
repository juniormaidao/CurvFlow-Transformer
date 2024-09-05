# CurvFlow-Transformer
conda create -n curvflowGT python=3.9
conda activate curvflowGT

bash install.sh

git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./

> cd examples/property_prediction/
> bash train.sh

