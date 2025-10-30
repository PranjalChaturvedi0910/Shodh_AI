# Policy Optimization for Financial Decision-Making

This repository contains the end-to-end machine learning solution for the **Shodh Hiring (ML) Project**. The project analyzes the LendingClub loan dataset to build and compare two different models for a fintech company's loan approval process:  
a **predictive Deep Learning (DL) model** and an **offline Reinforcement Learning (RL) agent**.

The objective is to go beyond simple default prediction and develop a policy that **maximizes financial return**, comparing a standard supervised model (Task 2) with a reward-maximizing offline RL policy (Task 3).

---

## 🚀 Project Tasks

This project consists of four main tasks:

1. **Task 1: Exploratory Data Analysis (EDA) & Preprocessing**  
   - The dataset is cleaned and analyzed for missing values.  
   - Feature engineering (e.g., mapping `emp_length`, creating `region`) and selection are performed.  
   - Encoding (Target Encoding) and scaling (StandardScaler) are applied.

2. **Task 2: Predictive Deep Learning Model**  
   - A Multi-Layer Perceptron (MLP) built using TensorFlow/Keras predicts the probability of loan default (binary classification).  
   - Model performance is evaluated using **AUC** and **F1-Score**.

3. **Task 3: Offline Reinforcement Learning Agent**  
   - The problem is reframed as an **offline RL** task:  
     - **State (s):** Preprocessed applicant features  
     - **Action (a):** {0 = Deny, 1 = Approve}  
     - **Reward (r):**  
       - Deny → 0 (no gain/loss)  
       - Approve (Paid) → + (loan_amnt × int_rate)  
       - Approve (Default) → − loan_amnt  
   - A `Discrete Conservative Q-Learning (CQL)` agent from **d3rlpy** learns an optimal approval policy from the static dataset.

4. **Task 4: Analysis & Comparison**  
   - Compare the predictive model’s implicit policy (e.g., “approve if default prob < 0.5”) with the RL agent’s explicit, reward-driven policy.  
   - Discuss results and insights in the **Final Report**.

---

## 📂 Repository Structure

```
.
├── input/
│   └── accepted_2007_to_2018Q4.csv       # (Download from Kaggle; not included)
├── pranjal-dl-eda.ipynb                  # Task 1 & Task 2: EDA + Deep Learning Model
├── reinforce-pranjal.ipynb               # Task 3: Offline RL Agent
├── requirements.txt                      # Python dependencies
├── Final_Report.pdf                      # Task 4: Analysis & Comparison
└── README.md                             # This file
```

---

## ⚙️ Setup & Installation

Follow these steps to reproduce results:

1. **Clone the Repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   cd YOUR_REPO_NAME
   ```

2. **Download the Data**
   - Get the dataset from [LendingClub Loan Data on Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club).
   - Create an `input/` folder in the project root.  
   - Place the CSV file inside it.  
   - Update these paths if needed:  
     - `DATA_FILE_PATH` in `pranjal-dl-eda.ipynb`  
     - `RAW_PATH` in `reinforce-pranjal.ipynb`

3. **Create a Virtual Environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   **Dependencies include:**
   - pandas  
   - numpy  
   - tensorflow  
   - scikit-learn==1.5.2  
   - imbalanced-learn  
   - category_encoders  
   - d3rlpy==2.4.0  
   - gymnasium[classic-control]==0.29.1  
   - joblib

---

## 🏃 Execution: How to Run

### 1. Run EDA and Deep Learning Model
**Notebook:** `pranjal-dl-eda.ipynb`

Steps performed:
1. Loads raw data (sample of 100,000 rows for faster runs).  
2. Cleans, preprocesses, and engineers features.  
3. Applies undersampling to balance classes.  
4. Builds, trains, and evaluates the Keras MLP model.

**Key Outputs:**
- **AUC Score:** ~0.93  
- **F1-Score (for Defaulted=1):** ~0.75  
- **Classification Report:** Precision/Recall for both classes

---

### 2. Run Offline RL Agent
**Notebook:** `reinforce-pranjal.ipynb`

Steps performed:
1. Loads raw data and applies minimal preprocessing.  
2. Defines a custom reward function as specified.  
3. Builds the offline dataset (`MDPDataset`) with transitions for both actions (Approve/Deny).  
4. Trains a `DiscreteCQL` agent.  
5. Evaluates policy by estimating average reward per loan.

**Key Outputs:**
- **Average reward per application (Estimated Policy Value)**  
- **Approval Rate (Policy’s tendency to approve)**  
- **Decision vs Outcome Matrix** (Deny/Approve vs Paid/Default)

---

## 🧠 Insights
- The **DL model** focuses on minimizing default probability.  
- The **RL agent** directly optimizes financial reward, balancing risk and profit.  
- This approach demonstrates how **offline RL can learn optimal financial decision policies** without live deployment.

---

## 📜 License
This repository is distributed for educational and hiring evaluation purposes only.

---
