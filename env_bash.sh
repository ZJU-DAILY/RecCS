conda install pip
pip install -r requirements.txt
conda install -c conda-forge metis
conda install -c conda-forge pymetis
pip install omegaconf
pip install pytorch-lightning
pip install overrides
pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
pip install dgl -f https://data.dgl.ai/wheels/torch-2.2/cu121/repo.html