# Titanic - Machine Learning from Disaster

This is a solution for the [Titanic: Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic) challenge on Kaggle. The goal is to build a predictive model that answers the question: **"what sorts of people were more likely to survive?"** using passenger data like age, gender, class, and more.

---


---

## 🧠 Approach

1. **Data Exploration & Cleaning**
   - Checked for missing values and outliers
   - Filled missing `Age`, `Embarked`, and `Fare`
   - Encoded categorical variables like `Sex` and `Embarked`

2. **Feature Engineering**
   - Created new features: `FamilySize`, `Title`, `IsAlone`
   - Binned `Age` and `Fare` into categories

3. **Modeling**
   - Tested multiple models: Logistic Regression, Random Forest, XGBoost
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

## 🧪 Models & Performance

| Model               | CV Accuracy   |
|--------------------|---------------|
| Logistic Regression| ~79%          |
| Random Forest      | ~81%          |
| XGBoost            | ~82%          |

> Final submission used **XGBoost** with optimized parameters.

---

## 📦 Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```


