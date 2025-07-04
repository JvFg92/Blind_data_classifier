# 🚀 Blind Data Classifier

This project provides a robust framework for classifying blind datasets, offering data preparation, feature selection, and the application of various machine learning classifiers. It's designed to streamline the process from raw data to trained and evaluated models.

## 🌟 Features

* **Automated Data Selection**: Automatically identifies the "best" suitable dataset for classification from a directory of CSV files based on criteria like unique classes, number of rows, and class balance.
* **Comprehensive Data Preprocessing**: Handles missing values, removes duplicate rows, and normalizes numerical features to ensure data quality.
* **Recursive Feature Elimination (RFE)**: Employs RFE with Logistic Regression to select the most relevant features, reducing dimensionality and improving model performance.
* **Flexible Data Splitting**: Splits data into training and testing sets for model development and evaluation.
* **Hyperparameter Tuning with GridSearchCV**: Utilizes the `train_and_tune_classifier` function to optimize classifier performance by performing a grid search over specified hyperparameters, using StratifiedKFold cross-validation and weighted F1 score to handle class imbalance effectively.
* **Multiple Classifiers**: Implements and evaluates popular classification algorithms:
    * K-Nearest Neighbors (KNN) 🏘️
    * Random Forest 🌳
    * Multi-layer Perceptron (MLP) / Neural Network 🧠
* **Detailed Classifier Evaluation**: Provides comprehensive evaluation metrics including Confusion Matrix, Classification Report, Accuracy, F1 Score, Precision, and Recall.
* **Insightful Visualizations**: Generates plots for data distribution, feature selection, train/test data distribution, and decision boundaries (for 2D data).


├── Blind_data_classifier/

│   └── bases/             # Place your raw CSV datasets here

│       ├── 01.csv

│       ├── 02.csv

│       └── ...

│   └── processed_data.csv # Processed data will be saved here

├── main.py                # Main script to run the classification pipeline

├── data_preparation.py    # Functions for data import, preprocessing, splitting, and feature selection

├── data_selection.py      # Logic for selecting the best database from the 'bases' folder

├── classifiers.py         # Functions for training and evaluating different classifiers

└── plot.py                # Utilities for generating various plots

### 🔍 Curiosity:
The databases used here are actually the same databases used in the Research article - Clustering cancer gene expression data: a comparative study. Available at the link:

https://link.springer.com/article/10.1186/1471-2105-9-497

## 🛠️ Installation - 🌐 Setting up your environment: 

### 0. Pre steps:
**On Linux 🐧:**
```bash
sudo apt update
sudo apt install python3-venv python3-full
```

**On Windows 🪟:**
⚠️ *On PowerShell:* ⚠️
```bash
python --version
pip --version
```
*If these commands fail, you may need to reinstall Python, ensuring you check the "Add Python to PATH" option.*

### 1. Virtual enviroment creation Linux/Windows:
```bash
# Create and enter a directory for your project
mkdir Classify
cd Classify

# Create the virtual environment named 'venv'
python3 -m venv venv
```

### 🌎 2. Activate the Virtual Environment:
**On Linux 🐧:**
```bash
source venv/bin/activate
```

**On Windows 🪟:**
```bash
.\venv\Scripts\activate
```
*You will know it's active because (venv) will appear at the beginning of your terminal prompt.*

### 3. Install Libraries:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn 
```

### 🚀 Usage - 4. Clone the Repository:
⚠️ **Make sure that you still in the correct directory** ⚠️
```bash
git clone https://github.com/JvFg92/Blind_data_classifier
```

### 5. Running the scripts: ▶️

**On Linux 🐧:**
```bash
cd Classify
```

```bash
source venv/bin/activate
```

```bash
cd Blind_data_classifier
```

```bash
python main.py
```

**On Windows 🪟:**
```bash
cd Classify
```

```bash
.\venv\Scripts\activate
```

```bash
cd Blind_data_classifier
```

```bash
#py main.py
python main.py
```


### ⚠️ 6. When you're finished, you can deactivate the environment with a single command: ⚠

```bash
deactivate
```

```bash
exit
```
