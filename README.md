# Titanic - Machine Learning from Disaster

This is a solution for the [Titanic: Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic) challenge on Kaggle. The goal is to build a predictive model that answers the question: **"what sorts of people were more likely to survive?"** using passenger data like age, gender, class, and more.

---


---

## Approach

1. **Data Exploration & Cleaning**
   - Checked for missing values and outliers
   - Filled missing `Age`, `Embarked`, and `Fare`
   - Encoded categorical variables like `Sex` and `Embarked`

2. **Data Visualisation**
   - Sperman Correlation
   - seaborn plots
   

2. **Feature Engineering**
   - Created new features: `FamilySize`, `Title`, `IsAlone`
   - Binned `Age` and `Fare` into categories
   - Outlier Handling

3. **Modeling**
   - Tested multiple models: Logistic Regression , XGBoost
   - Used cross-validation for evaluation
   - Tuned hyperparameters for best results

4. **Submission**
   - Made predictions on test set and saved to `submission.csv`

---

## 📊 Key Features Used

- Pclass
- Sex
- Age
- SibSp
- Parch
- Fare
- Embarked
- Title (from Name)
- IsAlone
- FamilySize

---

## Models & Performance

| Model              | CV Accuracy   |
|--------------------|---------------|
| Logistic Regression| ~81.42%       |
| XGBoost            | ~83.46%       |

> Final submission used **XGBoost** with optimized parameters.

---

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```
## How to Run

1. Clone this repository  
2. Add `train.csv` and `test.csv` to the `data/` directory  
3. Open and run the notebooks in `notebooks/`  
4. Export predictions to CSV for Kaggle submission  

---

## Resources

- [Kaggle Titanic Competition](https://www.kaggle.com/competitions/titanic)  
- [Pandas Docs](https://pandas.pydata.org/)  
- [scikit-learn Docs](https://scikit-learn.org/)  
- [XGBoost Docs](https://xgboost.readthedocs.io/)  

---

## License

MIT License. Feel Free to contribute


