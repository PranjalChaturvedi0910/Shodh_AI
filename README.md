# Policy Optimization for Financial Decision-Making

This repository contains the end-to-end machine learning solution for the **Shodh Hiring (ML) Project**. The project involves analyzing the LendingClub loan dataset to build and compare two different models for a fintech company's loan approval process: a predictive Deep Learning (DL) model and an offline Reinforcement Learning (RL) agent.

The primary objective is to move beyond simple default prediction and develop a policy that aims to **maximize financial return**, comparing a standard supervised approach (Task 2) with a reward-maximizing offline RL policy (Task 3).

## 🚀 Project Tasks

This project is broken down into four core tasks as specified by the assignment:

1.  **Task 1: Exploratory Data Analysis (EDA) & Preprocessing:** The data is loaded, analyzed for missing values, and cleaned. Features are engineered (e.g., `emp_length` mapping, `region` creation) and selected. All preprocessing, encoding (Target Encoding), and scaling (StandardScaler) steps are performed.
2.  **Task 2: Predictive Deep Learning Model:** A Multi-Layer Perceptron (MLP) using TensorFlow/Keras is built to predict the probability of loan default (a binary classification task). The model's performance is evaluated using **AUC and F1-Score**.
3.  **Task 3: Offline Reinforcement Learning Agent:** The problem is reframed as an offline RL task.
    * **State (s):** The preprocessed vector of applicant features.
    * **Action (a):** A discrete space {0: Deny, 1: Approve}.
    * **Reward (r):** A function engineered to reflect financial return:
        * `Deny`: **0** (no gain, no loss).
        * `Approve (Paid)`: **+ (loan_amnt * int_rate)** (profit).
        * `Approve (Default)`: **- loan_amnt** (loss).
    * An agent is trained using the **Discrete Conservative Q-Learning (CQL)** algorithm from the `d3rlpy` library to learn an optimal approval policy from the static dataset.
4.  **Task 4: Analysis & Comparison:** The results from both models are used to provide a detailed analysis, comparing the predictive model's implicit policy (e.g., "approve if default prob < 0.5") with the RL agent's explicit, reward-driven policy. This analysis is detailed in the separate **Final Report**.

---

## 📂 Repository Structure
.
├── input/
│   └── accepted_2007_to_2018Q4.csv    # (Not included, must be downloaded)
├── pranjal-dl-eda.ipynb             # Notebook for Task 1 (EDA) & Task 2 (DL Model)
├── reinforce-pranjal.ipynb          # Notebook for Task 3 (Offline RL Agent)
├── requirements.txt                 # Python dependencies
├── Final_Report.pdf                 # Task 4 Analysis (Submitted separately)
└── README.md                        # This file
---

## ⚙️ Setup & Installation

Follow these steps to set up the environment and reproduce the results.

1.  **Clone the Repository:**
    ```bash
    git clone [LINK_TO_YOUR_GITHUB_REPO]
    cd [REPOSITORY_NAME]
    ```

2.  **Download the Data:**
    * Download the `accepted_2007_to_2018Q4.csv` file from the [LendingClub Loan Data on Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club).
    * Create an `input/` directory in the root of the project.
    * Place the downloaded CSV file into this directory. The notebooks are configured to look for the file at this path: `/kaggle/input/shodhh/accepted_2007_to_2018Q4.csv`.
    * **Note:** You must update the path variable `DATA_FILE_PATH` in `pranjal-dl-eda.ipynb` (Cell 23) and `RAW_PATH` in `reinforce-pranjal.ipynb` (Cell 1) to match your local file structure (e.g., `input/accepted_2007_to_2018Q4.csv`).

3.  **Create a Virtual Environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

4.  **Install Dependencies:**
    A `requirements.txt` file is provided, generated from the notebooks.
    ```bash
    pip install -r requirements.txt
    ```
    This file includes:
    * `pandas`
    * `numpy`
    * `tensorflow`
    * `scikit-learn==1.5.2` (version specified in notebook)
    * `imbalanced-learn`
    * `category_encoders`
    * `d3rlpy==2.4.0` (version specified in notebook)
    * `gymnasium[classic-control]==0.29.1` (version specified in notebook)
    * `joblib`

---

## 🏃 Execution: How to Run

Run the notebooks sequentially.

### 1. Task 1 & 2: EDA and Deep Learning Model

Run the `pranjal-dl-eda.ipynb` notebook from top to bottom.

* **What it does:**
    1.  Loads the raw data (sampling 100,000 rows for speed).
    2.  Performs all **Task 1** cleaning, preprocessing, and feature engineering (Cells 23-27).
    3.  Performs a stratified train-test split.
    4.  Applies undersampling (`RandomUnderSampler`) to the training set to balance the classes.
    5.  Applies feature selection (`VarianceThreshold` and a pre-defined list) (Cell 30).
    6.  Builds, trains, and evaluates the **Task 2** Keras DL model (Cell 31).
* **Key Output (Cell 31):**
    * **AUC Score:** 0.9317
    * **F1-Score (for Defaulted=1):** 0.75
    * **Classification Report:** Provides precision and recall for both classes.

### 2. Task 3: Offline Reinforcement Learning Agent

Run the `reinforce-pranjal.ipynb` notebook from top to bottom.

* **What it does:**
    1.  Loads the raw data (Cell 4). **Note:** This notebook re-runs a minimal preprocessing pipeline for self-containment.
    2.  Defines the custom reward function as per **Task 3** specifications (Cell 6).
    3.  Builds the offline `MDPDataset` for `d3rlpy` by creating transitions for both possible actions (Approve/Deny) for each loan application (Cell 6).
    4.  Initializes the `DiscreteCQL` algorithm (Cell 7).
    5.  Trains the agent on the prepared dataset (Cell 8).
    6.  Evaluates the trained policy on the held-out test set by calculating the **Estimated Policy Value** (average reward) (Cell 9).
* **Key Output (Cell 9):**
    * **Average reward per application (Estimated Policy Value):** The final business metric.
    * **Approval Rate:** The percentage of loans the RL agent's policy decides to approve.
    * **Decision vs. Outcome Matrix:** A confusion matrix showing the agent's decisions (Deny/Approve) against the actual ground-truth outcomes (Paid/Default).
