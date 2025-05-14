# 🛰️ Land Type Classification using Sentinel-2 Satellite Images

## 📌 Project Overview

This project focuses on leveraging **Deep Neural Networks (DNNs)** to classify different land types using **Sentinel-2 satellite imagery** from the European Space Agency. The goal is to develop a model capable of distinguishing between various land types such as:

- 🌾 Agriculture  
- 🌊 Water  
- 🏙️ Urban Areas  
- 🏜️ Desert  
- 🛣️ Roads  
- 🌳 Trees  

The model will be beneficial for applications like **urban planning, environmental monitoring, and resource management** by providing accurate land use classification insights.

## 📊 Dataset

- The project will use **open-source datasets** or generate custom datasets using tools like **QGIS**.
- Sentinel-2 provides **multispectral images**, making it ideal for land classification tasks.

## 🛠 Tools & Libraries

The following tools and libraries will be used to process images and train the classification model:

- **Pandas** 🐼 - Data manipulation
- **NumPy** 🔢 - Numerical computing
- **os** 📂 - File handling
- **cv2** 📸 - Image processing
- **Matplotlib** 📊 - Data visualization
- **TensorFlow** 🤖 - Deep Learning framework
- **streamlit** - Deployment

## 🚀 Project Workflow

1. **Data Collection**
   - Download Sentinel-2 imagery.
2. **Exploratory Data Analysis (EDA)**
   - Visualize land type distributions.
   - Show samples from each class
3. **Data Preprocessing**
   - Preprocess images (resizing, normalization, etc.).
   - Generate labeled datasets.
4. **Model Development**
   - Build a **DNN model** using TensorFlow.
   - Train and optimize the model.
   - Evaluate performance using accuracy metrics.
5. **Deployment & Results Interpretation**
   - Deploy the model using streamlit
   - Test the model on new images.
   - Visualize classification results.
   - Interpret findings for practical applications.

## 🔗 Conclusion

By utilizing **Deep Learning and Satellite Imagery**, this project aims to provide **accurate land type classification**, assisting in multiple real-world applications such as **deforestation tracking, disaster response, and land resource management**. 🌍📡

---
🔗 **Analyze Land. Classify Smart. Transform Insights into Actions.**
=======
# 🌍 EuroSAT Land Type Classification

A deep learning project for satellite imagery classification using the [EuroSAT RGB dataset](https://www.kaggle.com/code/swaroopsrisailam/eurosat-land-classification). This project is developed as a final submission for a computer vision and deep learning course.

---

## 📌 Project Overview

The goal of this project is to classify satellite images into one of ten land cover classes such as residential, river, forest, and others using a Convolutional Neural Network (CNN).

### 🧠 Key Features

- Supervised image classification using CNNs
- Trained on EuroSAT RGB imagery (64x64 resolution)
- Metrics: Accuracy, Precision, Recall, F1-Score
- Visual analysis via confusion matrix & training curves
- Modular, readable, and reproducible codebase

---

## 🗃️ Dataset

- **Source**: [EuroSAT RGB Dataset on Kaggle](https://www.kaggle.com/code/swaroopsrisailam/eurosat-land-classification)
- **Images**: 27,000+
- **Classes**:
  - Residential
  - Industrial
  - River
  - Forest
  - Sea/Lake
  - Highway
  - Pasture
  - Annual Crop
  - Permanent Crop
  - Herbaceous Vegetation

---

## 🏗️ Project Structure

```

eurosat-land-type-classification/
├── data/                    
│   └── EuroSAT/
├── notebooks/            
│   ├── 1\_data\_exploration.ipynb
│   ├── 2\_model\_training.ipynb
│   ├── 3\_evaluation.ipynb
├── models/                  
│   └── eurosat\_cnn\_model.h5
├── utils/                   
│   ├── data\_utils.py
│   └── model\_utils.py
├── main.py                   
├── requirements.txt
└── README.md

```

---

## ⚙️ Installation

### 🐍 Clone & Install

```bash
git clone https://github.com/your-username/eurosat-land-type-classification.git
cd eurosat-land-type-classification
pip install -r requirements.txt
````

### 🧪 Launch Notebooks

```bash
jupyter notebook
```

---

## 🧠 Model Architecture

*Input: 64x64 RGB images
*3 Convolutional layers with ReLU + MaxPooling
*Fully connected dense layers
*Softmax output layer for 10-class classification
*Optimizer: Adam
*Loss: Categorical Crossentropy

## 📚 References

*[EuroSAT Paper (Helber et al.)](https://arxiv.org/abs/1709.00029)

*[Original EuroSAT Dataset](https://github.com/phelber/eurosat)

*[EuroSAT on Kaggle](https://www.kaggle.com/code/swaroopsrisailam/eurosat-land-classification)
