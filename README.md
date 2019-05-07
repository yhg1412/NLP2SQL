# Execution Procetures
1. Create a virtual environment with python2.7, activate this virtual environment and install the dependencies using:
pip install -r requirements.txt
2. Run the following command for training a SQLNet using toy dataset with column attention mechanisms: 
python train.py --toy --ca
3. After above step has completed, optionally train another SQLNet with column attention as well as trainable embeddings on toy dataset using the following command:
python train.py --toy --ca --train_emb
4. Wait till training finishes.
4. For testing purposes, following command tests the trained SQLNet with column attention on toy dataset:
python test.py --toy --ca
5. Following command tests the trained SQLNet with column attention and trainable embeddings on toy dataset:
python test.py --toy --ca --train_emb

