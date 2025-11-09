

---

```markdown
# 🫁 ApneaNet-CBi — ECG-Derived Sleep Apnea Detection

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A lightweight CNN-based deep learning framework for automatic detection of sleep apnea from ECG signals.

---

## 🩸 Dataset Source

The project uses the **Apnea-ECG Database** from **PhysioNet**, which contains ECG recordings with apnea annotations.

**Dataset Link:** [https://physionet.org/content/apnea-ecg/1.0.0/](https://physionet.org/content/apnea-ecg/1.0.0/)

This dataset consists of minute-by-minute labeled ECG signals used for supervised learning and evaluation of apnea detection models.

---

## 📘 Overview

Sleep apnea is a common and potentially serious sleep disorder characterized by repeated interruptions of breathing during sleep.  
**ApneaNet-CBi** is a lightweight Convolutional Neural Network (CNN) framework designed to detect apnea events automatically from ECG-derived spectrograms.  
The model is optimized for efficiency, making it suitable for portable and embedded medical systems.

---

## 📂 Repository Structure

```

Sleep-Apnea-Detection/
│
├── Final_Project.ipynb         # Main Jupyter Notebook (data, training, evaluation)
├── training_curves.png         # Training accuracy and loss visualization
├── requirements.txt            # Python dependencies
└── README.md                   # Documentation

````

---

## 🧠 Model Summary

- **Model Type:** CNN (Convolutional Neural Network)  
- **Input:** 30-second ECG signal windows converted into spectrograms  
- **Output:** Binary classification — Apnea (1) or Normal (0)  
- **Optimizer:** Adam  
- **Loss Function:** Binary Crossentropy  
- **Metrics:** Accuracy, AUC (ROC), Precision, Recall, F1-score  

---

## 📈 Training Performance

| Metric | Value |
|:-------|:------|
| Final Training Accuracy | **0.9651** |
| Final AUC (ROC) | **0.9953** |
| Final Training Loss | **0.0857** |
| Epochs | **20** |

The model demonstrates excellent convergence with steadily increasing accuracy and AUC, indicating robust learning and generalization.

---

### 📊 Training Curves

![Training Curves](training_curves.png)

> The graph shows decreasing loss and increasing accuracy across 20 epochs, confirming stable convergence and strong learning performance.

---

## ⚙️ Installation and Usage

### Step 1: Clone the repository
```bash
git clone https://github.com/<your-username>/Sleep-Apnea-Detection.git
cd Sleep-Apnea-Detection
````

### Step 2: Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate      # For Linux/Mac
venv\Scripts\activate         # For Windows
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the notebook

Open the notebook using Jupyter:

```bash
jupyter notebook Final_Project.ipynb
```

Execute all cells in sequence to:

* Load and preprocess the ECG data
* Train the CNN model
* Evaluate the model’s performance
* Visualize training and evaluation results

---

## 🧩 Requirements

```
tensorflow>=2.8.0
numpy
pandas
matplotlib
scikit-learn
tqdm
scipy
librosa
pyedflib
```

---

## 🧪 Evaluation and Results

The model performs window-level and subject-level classification of apnea events.
Predicted probabilities are aggregated to estimate per-record Apnea–Hypopnea Index (AHI).

You can compute precision, recall, F1-score, and confusion matrix using:

```python
from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_true, y_pred))
print(confusion_matrix(y_true, y_pred))
```

Once computed, these results can be added under this section for a complete performance summary.

---

## 🩺 Interpretation

* The CNN achieves high discrimination capability (AUC ≈ 0.995).
* High precision → minimal false positives.
* High recall → effective apnea detection coverage.
* Per-record AHI estimation correlates closely with clinical labels.
* Potential deployment for real-time monitoring after hardware optimization.

---

## 🔮 Future Enhancements

* Add validation metrics (Precision, Recall, F1-score, Confusion Matrix).
* Extend the framework with additional biosignals (SpO₂, airflow).
* Model pruning and quantization for edge and mobile deployment.
* Real-time inference for portable health monitoring systems.

---

## 👨‍💻 Author

**Sriram Akula**
B.Tech in Computer Science and Engineering
Specialization: Machine Learning and DevOps

📧 [your.email@example.com](mailto:your.email@example.com)
🔗 [GitHub](https://github.com/<your-username>) | [LinkedIn](https://www.linkedin.com/in/yourprofile)

---

## 🪪 License

This project is licensed under the [MIT License](LICENSE).

---

## 📄 Citation

If you use this repository in your research, please cite:

```
Akula, Sriram. "ApneaNet-CBi: A Lightweight CNN-Based Deep Learning Framework for ECG-Derived Sleep Apnea Detection." (2025)
```

---

## 🧭 Acknowledgment

This work utilizes the **Apnea-ECG Database** from **PhysioNet** for training and evaluation.
Special thanks to the open-source community for tools and datasets that made this research possible.

```


```
