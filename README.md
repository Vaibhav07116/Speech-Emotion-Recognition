# Speech Emotion Recognition Using Deep Learning Techniques

This project presents a comprehensive comparative analysis of four distinct deep learning architectures for Speech Emotion Recognition (SER). The models—CNN+LSTM, Emoformer, Vision Transformer (ViT), and ResNet—are trained and evaluated on the TESS and RAVDESS datasets to determine the most effective and efficient approach for classifying human emotions from audio.

## 🧑‍💻 Authors

* **Jasmine Selvakumari Jeya I** (Senior Associate Professor)
* **Deepansh Chandra**
* **Abhinav Singh**
* **Vaibhav**
* **Manim Rohit Rao**
* **Tejas Gaur**

## ✨ Key Features

* Implementation of four state-of-the-art deep learning models for SER.
* Experiments conducted on both the **TESS dataset** and a combined **TESS+RAVDESS** dataset.
* Feature extraction using **Mel-spectrograms**.
* Complete scripts for training, evaluation (including Confusion Matrix and ROC/AUC curves), and prediction for each experiment.
* In-depth comparative analysis of model performance vs. computational efficiency.

## 🧠 Models Implemented

1.  **CNN+LSTM:** A hybrid model combining Convolutional and Recurrent layers.
2.  **Emoformer:** A Transformer-based architecture tailored for speech processing.
3.  **Vision Transformer (ViT):** A Transformer architecture that treats spectrograms as images.
4.  **ResNet:** A deep Convolutional Neural Network with residual connections.

## ⚙️ Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Vaibhav07116/Speech-Emotion-Recognition.git](https://github.com/Vaibhav07116/Speech-Emotion-Recognition.git)
    cd Speech-Emotion-Recognition
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv .venv
    # On Windows
    .venv\Scripts\activate
    # On macOS/Linux
    source .venv/bin/activate
    ```

3.  **Install the required libraries:**
    ```bash
    pip install torch torchaudio pandas librosa scikit-learn matplotlib seaborn tqdm
    ```

## 🚀 Usage

1.  **Place Datasets:** Create a main `Dataset` folder in the project root. Place your `TESS` and `ravdess` folders inside it.
2.  **Navigate to an Experiment:** The project is organized by dataset and model. Choose an experiment to run, for example:
    ```bash
    # To run Emoformer on the combined dataset
    cd combined/emoformer
    ```
3.  **Train a Model:** Run the training script from within the experiment's folder:
    ```bash
    python train.py
    ```
4.  **Evaluate a Model:** After training is complete, run the evaluation script:
    ```bash
    python evaluate_model.py
    ```

