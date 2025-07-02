
## 🌐 Setting up your environment 

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

###  2. Activate the Virtual Environment:
**On Linux 🐧:**
```bash
source venv/bin/activate
```

**On Windows 🪟:**
```bash
.\venv\Scripts\activate
```
*You will know it's active because (venv) will appear at the beginning of your terminal prompt.*

### 3. Install Spade:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn 
```

### 4. Clone the Repository:
⚠️ **Make sure that you still in the correct directory** ⚠️
```bash
git clone https://github.com/JvFg92/Blind_data_classifier
```

### 5. Running the scripts: ▶️
**Generator:**
⚠️ **Open a Terminal** ⚠️

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


### 6. When you're finished, you can deactivate the environment with a single command:
⚠️ **Do it for each terminal** ⚠️
```bash
deactivate
```

```bash
exit
```
