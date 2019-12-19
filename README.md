# Pokemon generation using Generative Adversarial Network
This is the assignment 4 for CMSC5707 - Advance Topics in Artificial Intelligence.

## Dataset
1. MNIST from Tensorflow
2. Pokedex from https://github.com/PokeAPI/sprites/tree/master/sprites/pokemon

## Model
Deep Convolutional Generative Adversarial Network

## How to run
1. Install dependencies
```
pip3 install -r requirements.txt
```

2. Train GAN for MNIST
```
# 100 epochs, generate 16 images at once, batch size = 256 
python3 mnistGAN.py 100 16 256
```

3. Train GAN for Pokedex
```
git clone https://github.com/PokeAPI/sprites.git

cp ./sprites/pokemon .

python3 preprocess.py

# 3000 epochs, generate 16 images at once, batch size = 8, noise dimension = 128 
python3 pokeGAN.py 3000 16 8 128
```
