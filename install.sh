#!/usr/bin/env bash
# command to install this enviroment: source init.sh

# install miniconda3 if not installed yet.
#wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
#bash Miniconda3-latest-Linux-x86_64.sh
#source ~/.bashrc

# install PyTorch
conda create -n denet -y python=3.10 numpy=1.24 numba
conda activate denet

# please always double check installation for pytorch and torch-scatter from the official documentation
pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
conda install -y cudatoolkit=11.7 -c nvidia
pip install torch-scatter==2.1.1 -f https://data.pyg.org/whl/torch-2.0.1+cu117.html

pip install -r requirements.txt

# install cpp extensions, the pointnet++ library
cd openpoints/cpp/pointnet2_batch
python setup.py install
cd ../

# grid_subsampling library. necessary only if interested in S3DIS_sphere
# cd subsampling
# python setup.py build_ext --inplace
# cd ..


# # point transformer library. Necessary only if interested in Point Transformer and Stratified Transformer
# cd pointops/
# python setup.py install
# cd ..

# Blow are functions that optional. Necessary only if interested in reconstruction tasks such as completion
# cd chamfer_dist
# python setup.py install --user
# cd ../emd
# python setup.py install --user
# cd ../../../





# #!/usr/bin/env bash
# # command to install this enviroment: source init.sh

# # install miniconda3 if not installed yet.
# #wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
# #bash Miniconda3-latest-Linux-x86_64.sh
# #source ~/.bashrc

# # install PyTorch
# conda create -n denet -y python=3.10 numpy=1.24 numba
# conda activate denet

# # please always double check installation for pytorch and torch-scatter from the official documentation
# conda install -y pytorch=2.0.1 torchvision=0.15.2 cudatoolkit=11.7 -c pytorch -c nvidia
# pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.1+cu117.html

# pip install -r requirements.txt

# # install cpp extensions, the pointnet++ library
# cd openpoints/cpp/pointnet2_batch
# python setup.py install
# cd ../

# # grid_subsampling library. necessary only if interested in S3DIS_sphere
# # cd subsampling
# # python setup.py build_ext --inplace
# # cd ..


# # # point transformer library. Necessary only if interested in Point Transformer and Stratified Transformer
# # cd pointops/
# # python setup.py install
# # cd ..

# # Blow are functions that optional. Necessary only if interested in reconstruction tasks such as completion
# # cd chamfer_dist
# # python setup.py install --user
# # cd ../emd
# # python setup.py install --user
# # cd ../../../