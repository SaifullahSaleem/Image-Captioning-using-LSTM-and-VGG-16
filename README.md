# Image Captioning using LSTM and VGG16

## Project Overview
This project implements an **Image Captioning System** using **VGG16 for feature extraction** and **LSTM for sequence generation**. The system takes an image as input and generates a textual caption describing the content of the image. The dataset used is **Flickr8k**, which contains images with associated captions.

## Features
- Uses **VGG16** (pre-trained on ImageNet) to extract deep features from images.
- Applies **Tokenization** and **Sequence Padding** for text preprocessing.
- Implements **LSTM-based Decoder** to generate captions from extracted features.
- Trains on **Flickr8k Dataset** with captions.

## Dataset
- **Flickr8k Dataset**: Contains 8,000 images with five captions per image.
- Image features are extracted using **VGG16** and stored for training the captioning model.


## How to Run
1. **Feature Extraction**
   - Load **VGG16** and extract features for all images.
   - Save extracted features for later use.

2. **Data Preprocessing**
   - Tokenize captions and create word mappings.
   - Pad sequences to uniform length.

3. **Train the Model**
   - Use **LSTM-based decoder** to learn image-caption mapping.
   - Save the trained model.

4. **Generate Captions for New Images**
   - Load the trained model.
   - Extract image features using **VGG16**.
   - Predict the caption using the LSTM model.

## Model Architecture
- **VGG16 Feature Extractor**: Pre-trained convolutional model extracts image embeddings.
- **LSTM-based Decoder**: Generates captions based on extracted image features.
- **Embedding Layer**: Maps words to dense vectors.
- **Fully Connected Layer**: Final output layer to generate words sequentially.

## Results & Evaluation
- The model generates captions by learning from the dataset.
- It can describe images with reasonable accuracy.
- BLEU Score or other evaluation metrics can be used to measure accuracy.



