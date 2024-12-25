# Plant Disease Prediction Model

Welcome to the **Plant Disease Prediction Model** repository! This project leverages Convolutional Neural Networks (CNNs) and Transfer Learning to identify diseases in plants from leaf images. Built using a real-world dataset from Kaggle, the model aims to aid in early detection and management of plant diseases.

## 🌟 Features

- **Deep Learning Architecture**: Uses CNNs for feature extraction and Transfer Learning for enhanced accuracy.
- **Real Dataset**: Trained on a high-quality dataset sourced from Kaggle.
- **Scalable**: Easily adaptable to new datasets or additional plant species.
- **User-Friendly**: Simplified pipeline for training, testing, and deploying the model.

## 🚀 Getting Started

Follow these steps to set up and use the model:

### Prerequisites

Ensure you have Python installed along with the required libraries:

```bash
pip install tensorflow numpy pandas matplotlib scikit-learn
```

### Installation

Clone this repository to your local machine:

```bash
git clone https://github.com/your-username/plant-disease-prediction.git
cd plant-disease-prediction
```

### Dataset Preparation

1. Download the dataset from Kaggle: [Plant Disease Dataset](https://www.kaggle.com/)
2. Extract the dataset into the `data/` folder.
   - The folder structure should look like:
     ```
     data/
     ├── Train/
     │   ├── Apple_Scab/
     │   ├── Healthy/
     │   └── ...
     └── Validation/
         ├── Apple_Scab/
         ├── Healthy/
         └── ...
     ```

### Model Training

Train the model by running:

```bash
python train_model.py
```

This script will:
- Preprocess the dataset
- Use a CNN with Transfer Learning (e.g., ResNet, VGG16, or EfficientNet)
- Save the trained model in the `model/` directory

### Model Evaluation

Evaluate the model's performance on the validation set:

```bash
python evaluate_model.py
```

This script outputs:
- Accuracy
- Precision, Recall, and F1 Score
- Confusion Matrix

### Prediction

Use the model to predict plant diseases on new images:

```bash
python predict.py --image path/to/leaf_image.jpg
```

Example output:
```
Predicted class: Healthy
Confidence: 98.6%
```

## 📂 Project Structure

```
.
├── data
│   ├── Train/                  # Training dataset
│   └── Validation/             # Validation dataset
├── model
│   └── plant_disease_model.h5  # Trained CNN model
├── train_model.py              # Training script
├── evaluate_model.py           # Evaluation script
├── predict.py                  # Prediction script
├── requirements.txt            # Dependencies
├── README.md                   # Project documentation
```

## 🔬 How It Works

1. **Data Preprocessing**:
   - Resizes images to uniform dimensions.
   - Augments data to enhance training robustness.
   - Splits dataset into training and validation sets.

2. **Model Architecture**:
   - Implements Transfer Learning using pre-trained CNN models (e.g., ResNet50).
   - Fine-tunes the model on plant disease data.

3. **Prediction**:
   - Takes input as a leaf image.
   - Outputs the predicted disease class and confidence score.

## 📊 Results

- **Accuracy**: Achieved over 90% accuracy on the validation set.
- **Confusion Matrix**: Visualizes model performance across classes.

## 🧪 Testing

You can test the model with custom images or extend the dataset for better generalization. Use the `predict.py` script for real-time predictions.

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch-name`).
3. Commit your changes (`git commit -m "Add some feature"`).
4. Push to the branch (`git push origin feature-branch-name`).
5. Open a Pull Request.

## 📖 References

- TensorFlow Documentation: [https://www.tensorflow.org/](https://www.tensorflow.org/)
- Kaggle Dataset: [https://www.kaggle.com/](https://www.kaggle.com/)
- Transfer Learning Guide: [https://keras.io/guides/transfer_learning/](https://keras.io/guides/transfer_learning/)


## 📧 Contact

For any inquiries, please reach out to:
- **Satyam Singh**
- Email: [satyamsingh7734@gmail.com

---

### 🌟 If you find this project useful, please give it a ⭐ on GitHub!
