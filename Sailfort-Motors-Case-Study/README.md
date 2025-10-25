[Jupyter Notebook Binder](https://mybinder.org/v2/gh/Ignatimus/data-analysis-portfolio/HEAD?urlpath=%2Fdoc%2Ftree%2FSailfort-Motors-Case-Study%2FCapstone+Project+Salifort+Motors+Jupyter+Notebook.ipynb) 

## Tools Used
* JupyterLab
* Python
* Numpy
* Pandas
* Matplotlib (Pyplot)
* Seaborn
* Sklearn (.metrics, .linear_model, .tree, .model_selection, .ensamble, .xgboost)

## Business Problem
Salifort Motors seeks to improve employee retention and answer the following question:
**What’s likely to make an employee leave the company?**

## Key Insights from Data Analysis
1. Two groups of employees tend to leave:
    * **Underworked employees** – possibly dismissed due to low output.
    * **Overworked employees** – likely resigned from burnout.
2. Employees with 7 projects all left the company.
3. Employees working 250+ hours per month showed extreme workloads and low satisfaction.
4. Employees who survive past year 6 rarely leave.
5. The optimal workload appears to be 3–4 projects per employee.
6. `Satisfaction level`, `number of projects`, `evaluation score`, and `time at company` are key predictors of turnover.

## Model Development and Evaluation
Five models were trained and compared:
* **Logistic** Regression – High recall but many false positives.
* **Decision** Tree (untuned) – Overfitted the data.
* **Tuned Decision Tree** – Improved performance, reduced overfitting.
* **Random Forest** – Best generalization and stable performance.
* **XGBoost** – Highest accuracy but mild overfitting.
Across all datasets, **Random Forest** achieved the best balance between accuracy and generalization, followed closely by **XGBoost**.

## Recommendations
1. Limit the number of projects assigned per employee.
2. Investigate dissatisfaction among employees with ~4 years of tenure.
3. Reward or compensate long working hours, or reduce workload expectations.
4. Clarify company policies regarding overtime and workload.
5. Encourage open communication about workload and recognition.
6. Adjust evaluation criteria to avoid rewarding only overworked employees.
