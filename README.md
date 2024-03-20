---
jupyter:
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
  language_info:
    codemirror_mode:
      name: ipython
      version: 3
    file_extension: .py
    mimetype: text/x-python
    name: python
    nbconvert_exporter: python
    pygments_lexer: ipython3
    version: 3.11.3
  nbformat: 4
  nbformat_minor: 2
---

::: {.cell .markdown}
### Assignment 1 SEA 600 Technical Report

-   Fahad Ali Khan
-   Abhi NileshKumar Patel
-   Inderpreet Singh Parmar
:::

::: {.cell .markdown}
### Problem & Data Description {#problem--data-description}

-   Objective: The project aims to predict whether a loan will be
    approved using historical lending data from LendingClub, a
    peer-to-peer lending company. This is a binary classification
    problem where the outcome is whether a loan is approved or not. See:
    <https://figshare.com/articles/dataset/Lending_Club/22121477?file=39316160>

-   ML Problem Translation: The objective translates into a supervised
    machine learning problem where the goal is to classify loan
    applications as approved or not approved based on features extracted
    from historical data.

#### Implementation Constraints:

##### Resource Utilization: Models need to be resource-efficient due to constraints on computation time and memory. This affects the choice of algorithms, favoring those with lower complexity. {#resource-utilization-models-need-to-be-resource-efficient-due-to-constraints-on-computation-time-and-memory-this-affects-the-choice-of-algorithms-favoring-those-with-lower-complexity}

##### Societal Impact: The model\'s predictions could impact individuals\' financial opportunities, necessitating high accuracy and fairness in predictions to avoid discriminatory outcomes. {#societal-impact-the-models-predictions-could-impact-individuals-financial-opportunities-necessitating-high-accuracy-and-fairness-in-predictions-to-avoid-discriminatory-outcomes}

##### Regulatory Compliance: Adherence to financial regulations such as the Equal Credit Opportunity Act (ECOA) is necessary to ensure non-discriminatory lending practices. {#regulatory-compliance-adherence-to-financial-regulations-such-as-the-equal-credit-opportunity-act-ecoa-is-necessary-to-ensure-non-discriminatory-lending-practices}
:::

::: {.cell .markdown}
### Data Description and Data preprocessing

-   Our Target Variable is *Loan_Status* that is used to predict if a
    persons loan request is accepted or rejected.

  -----------------------------------------------------------------------
  Feature Name                                      Description
  ------------------------------------------------- ---------------------
  issue_d                                           The month and year
                                                    when the loan was
                                                    funded, indicating
                                                    when the loan
                                                    agreement started.

  sub_grade                                         A granular
                                                    categorization within
                                                    a broader credit
                                                    grade, providing a
                                                    more detailed
                                                    assessment of credit
                                                    risk.

  term                                              The duration of the
                                                    loan term, typically
                                                    in months, affecting
                                                    monthly payments and
                                                    total interest.

  home_ownership                                    Indicates the
                                                    borrower\'s housing
                                                    situation (owning,
                                                    renting, etc.), which
                                                    can impact
                                                    creditworthiness.

  fico_range_low                                    The lower end of the
                                                    borrower\'s FICO
                                                    score range at
                                                    application, used to
                                                    evaluate credit risk.

  total_acc                                         Total number of
                                                    credit lines in the
                                                    borrower\'s credit
                                                    history, reflecting
                                                    credit experience and
                                                    utilization.

  pub_rec                                           Number of derogatory
                                                    public records on the
                                                    borrower\'s credit
                                                    report, affecting
                                                    creditworthiness.

  revol_util                                        Percentage of
                                                    revolving credit used
                                                    by the borrower,
                                                    indicating credit
                                                    utilization and
                                                    potential risk.

  annual_inc                                        The borrower\'s
                                                    self-reported annual
                                                    income, crucial for
                                                    assessing loan
                                                    repayment ability.

  int_rate                                          The interest rate on
                                                    the loan, directly
                                                    affecting the cost of
                                                    borrowing and monthly
                                                    payments.

  purpose                                           The self-reported
                                                    reason for the loan,
                                                    providing context on
                                                    its intended use.

  mort_acc                                          Number of mortgage
                                                    accounts, which can
                                                    signify financial
                                                    stability and credit
                                                    history.

  loan_amnt                                         The applied loan
                                                    amount, influencing
                                                    the borrower\'s debt
                                                    obligations and
                                                    repayment terms.

  application_type                                  Indicates if the
                                                    application is
                                                    individual or joint,
                                                    affecting credit
                                                    assessment and
                                                    repayment
                                                    responsibility.

  installment                                       Monthly payment owed
                                                    if the loan
                                                    originates, based on
                                                    amount, term, and
                                                    rate.

  verification_status                               Status of income and
                                                    employment
                                                    verification,
                                                    impacting perceived
                                                    loan risk.

  pub_rec_bankruptcies                              Number of bankruptcy
                                                    public records,
                                                    significantly
                                                    affecting
                                                    creditworthiness.

  addr_state                                        The borrower\'s state
                                                    of residence, useful
                                                    for geographic risk
                                                    analysis and
                                                    regulatory
                                                    compliance.

  initial_list_status                               Indicates the loan\'s
                                                    market status,
                                                    affecting liquidity
                                                    and pricing.

  fico_range_high                                   The higher end of the
                                                    borrower\'s FICO
                                                    score range,
                                                    providing a fuller
                                                    picture of credit
                                                    standing.

  revol_bal                                         Total balance on
                                                    revolving accounts,
                                                    indicating credit
                                                    utilization and
                                                    financial management.

  id                                                Unique identifier for
                                                    the loan or borrower,
                                                    essential for data
                                                    tracking.

  open_acc                                          Number of open credit
                                                    lines, showing credit
                                                    usage and
                                                    availability.

  emp_length                                        Employment duration
                                                    at current job,
                                                    indicating job
                                                    stability and income
                                                    reliability.

  loan_status                                       Current status of the
                                                    loan, crucial for
                                                    loan performance
                                                    assessment.

  time_to_earliest_cr_line                          Time since the first
                                                    credit line was
                                                    opened, reflecting
                                                    credit history
                                                    length.
  -----------------------------------------------------------------------

#### The Dataset is divided into two with the Test set 90k records and Traning set with 235K records
:::

::: {.cell .markdown}
Loading necessary Libraries
:::

::: {.cell .code execution_count="1"}
``` python
%pip install -q hvplot
# Install memory_profiler 
%pip install memory_profiler
%pip install imbalanced-learn

# Load memory_profiler extension
%load_ext memory_profiler

import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
import hvplot.pandas
from IPython.display import display
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix,  roc_curve, auc, precision_recall_curve, average_precision_score,RocCurveDisplay
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import make_scorer, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.discriminant_analysis import  LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import time

pd.set_option('display.float', '{:.2f}'.format)
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 50)
```

::: {.output .stream .stdout}
    Note: you may need to restart the kernel to use updated packages.
:::

::: {.output .stream .stderr}

    [notice] A new release of pip is available: 23.3.2 -> 24.0
    [notice] To update, run: python.exe -m pip install --upgrade pip
:::

::: {.output .stream .stdout}
    Requirement already satisfied: memory_profiler in c:\users\fahad\appdata\local\programs\python\python311\lib\site-packages (0.61.0)
    Requirement already satisfied: psutil in c:\users\fahad\appdata\roaming\python\python311\site-packages (from memory_profiler) (5.9.8)
    Note: you may need to restart the kernel to use updated packages.
:::

::: {.output .stream .stderr}

    [notice] A new release of pip is available: 23.3.2 -> 24.0
    [notice] To update, run: python.exe -m pip install --upgrade pip

    [notice] A new release of pip is available: 23.3.2 -> 24.0
    [notice] To update, run: python.exe -m pip install --upgrade pip
:::

::: {.output .stream .stdout}
    Requirement already satisfied: imbalanced-learn in c:\users\fahad\appdata\local\programs\python\python311\lib\site-packages (0.12.0)
    Requirement already satisfied: numpy>=1.17.3 in c:\users\fahad\appdata\local\programs\python\python311\lib\site-packages (from imbalanced-learn) (1.24.2)
    Requirement already satisfied: scipy>=1.5.0 in c:\users\fahad\appdata\local\programs\python\python311\lib\site-packages (from imbalanced-learn) (1.10.1)
    Requirement already satisfied: scikit-learn>=1.0.2 in c:\users\fahad\appdata\local\programs\python\python311\lib\site-packages (from imbalanced-learn) (1.4.1.post1)
    Requirement already satisfied: joblib>=1.1.1 in c:\users\fahad\appdata\local\programs\python\python311\lib\site-packages (from imbalanced-learn) (1.3.2)
    Requirement already satisfied: threadpoolctl>=2.0.0 in c:\users\fahad\appdata\local\programs\python\python311\lib\site-packages (from imbalanced-learn) (3.2.0)
    Note: you may need to restart the kernel to use updated packages.
:::

::: {.output .display_data}
``` json
""
```
:::

::: {.output .display_data}
``` json
""
```
:::

::: {.output .display_data}
```{=html}
<style>*[data-root-id],
*[data-root-id] > * {
  box-sizing: border-box;
  font-family: var(--jp-ui-font-family);
  font-size: var(--jp-ui-font-size1);
  color: var(--vscode-editor-foreground, var(--jp-ui-font-color1));
}

/* Override VSCode background color */
.cell-output-ipywidget-background:has(
    > .cell-output-ipywidget-background > .lm-Widget > *[data-root-id]
  ),
.cell-output-ipywidget-background:has(> .lm-Widget > *[data-root-id]) {
  background-color: transparent !important;
}
</style>
```
:::

::: {.output .display_data}
``` json
""
```
:::
:::

::: {.cell .code execution_count="2"}
``` python
data = pd.read_csv("train_lending_club.csv")
print("Data imported successfully")
test_data = pd.read_csv("test_lending_club.csv")
print("Test data imported successfully")

#filling Test Data empty values  Training data has no empty records

numerical_cols = test_data.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = test_data.select_dtypes(include=['object', 'category']).columns

# Impute numerical columns with the median
num_imputer = SimpleImputer(strategy='median')
test_data[numerical_cols] = num_imputer.fit_transform(test_data[numerical_cols])

# Impute categorical columns with the most frequent value (mode)
cat_imputer = SimpleImputer(strategy='most_frequent')
test_data[categorical_cols] = cat_imputer.fit_transform(test_data[categorical_cols])

#One-Hot Encoding
# List of categorical columns to convert
categorical_columns = ['sub_grade', 'term', 'home_ownership', 'purpose', 'application_type', 'verification_status', 'initial_list_status']

# One-hot encode these columns
data = pd.get_dummies(data, columns=categorical_columns,drop_first=True)
test_data = pd.get_dummies(test_data, columns=categorical_columns, drop_first=True)

# Ensure that both dataframes have the same dummy columns
data, test_data = data.align(test_data, join='left', axis=1, fill_value=0)


# Store 'loan_status' in a separate variable and then drop non-feature columns from the training data
y_train = data['loan_status']
X_train = data.drop(['issue_d', 'loan_status', 'id', 'addr_state'], axis=1)

y_test = test_data['loan_status']
X_test = test_data.drop(['issue_d', 'loan_status', 'id', 'addr_state'], axis=1)

smote = SMOTE(sampling_strategy='minority', random_state=42)  
# Fit SMOTE on Training Data
X_train, y_train= smote.fit_resample(X_train, y_train)
# Fit SMOTE on Testing Data
X_test, y_test = smote.fit_resample(X_test, y_test)

# Assuming `X_test` and `y_test` are your existing test features and labels
# Split the test set into a smaller test set and a validation set
X_test_smaller, X_validation, y_test_smaller, y_validation = train_test_split(
    X_test, y_test, test_size=0.5, random_state=42
)
```

::: {.output .stream .stdout}
    Data imported successfully
    Test data imported successfully
:::
:::

::: {.cell .markdown}
#### We now have Clean processed data

-   Did you need to clean the data? Why and how? What were your options?
    Why did you choose your method(s)?
-   (ANS) Yes we cleaned the data by
    -   Removing imbalance as the majority class was too high making the
        models not being able to capture the complexity of the data. We
        also removed issue_id feature since it conflicted with our ML
        Problem the model shouldn\'t know the date of the loan being
        rejected/accepted. Id and addr_state features were also removed
        as they were deemed invalid.
-   How did you split the data? What were your options? Why did you
    choose your method(s)?
-   (ANS) The test data were splitted into half to give validation set.
    This was done to follow assignment instruction and use test data to
    only check final accuracy of the models.
:::

::: {.cell .markdown}
## Milestone I and II

-   Selected Models : Logistic Regression, KNN, Linear Discriminant
    Analysis and Decision Trees
-   Metrics use to evaluate ROC_AUC, PR_AUC, F1_Score, classification
    report.
-   *Alternatives Considered* SVM: Since it is good with high
    dimensional data. However not used because of the large training
    dataset.
-   See Code for CV of these models.
-   See apendix for ROC and PR curves of the CV of the models
:::

::: {.cell .markdown}
-   Here is the table summarizing the performance and resource
    utilization of the models on the **training dataset**:

  --------------------------------------------------------------------------------------
  Model        Precision-Recall   ROC AUC F1      Training Time Peak       Memory
               AUC                        Score   (seconds)     Memory     Increment
                                                                (MiB)      (MiB)
  ------------ ------------------ ------- ------- ------------- ---------- -------------
  LDA          0.94               0.776   0.901   2.781         1236.81    732.88

  Decision     0.91               0.552   0.810   16.527        583.27     76.55
  Tree                                                                     

  Logistic     0.94               0.779   0.903   1.023         824.35     147.03
  Regression                                                               

  KNN          0.93               0.694   0.895   0.094         820.09     142.79
  --------------------------------------------------------------------------------------
:::

::: {.cell .markdown}
1.  **Precision-Recall AUC**:
    -   **LDA** and **Logistic Regression** have the highest
        Precision-Recall AUC scores at **0.94**, indicating they perform
        well in terms of both precision (how many selected items are
        relevant) and recall (how many relevant items are selected).
    -   **KNN** follows closely with a Precision-Recall AUC of **0.93**.
    -   **Decision Tree** has the lowest score at **0.91**, which is
        still quite respectable but indicates it might not perform as
        well as the other models in distinguishing between the classes,
        especially in imbalanced datasets.
2.  **ROC AUC**:
    -   **Logistic Regression** leads slightly with a ROC AUC of
        **0.779**, indicating its ability to distinguish between the
        classes is slightly better than the others.
    -   **LDA** is a close second with **0.776**.
    -   **KNN** and **Decision Tree** have lower ROC AUC scores of
        **0.694** and **0.552** respectively, indicating they may not
        distinguish as effectively between the classes as the logistic
        regression and LDA models.
3.  **F1 Score**:
    -   Both **LDA** and **Logistic Regression** show high F1 scores of
        **0.901** and **0.903**, suggesting a strong balance between
        precision and recall.
    -   **KNN** has a slightly lower F1 score of **0.895**.
    -   **Decision Tree** has the lowest F1 score at **0.810**,
        indicating it may not balance false positives and false
        negatives as well as the other models.
4.  **Training Time**:
    -   **KNN** is the fastest model to train with a time of **0.094
        seconds**, making it highly efficient for training purposes.
    -   **Logistic Regression** also shows impressive efficiency with a
        training time of **1.023 seconds**.
    -   **LDA** takes slightly longer at **2.781 seconds**.
    -   **Decision Tree** takes the longest to train at **16.527
        seconds**, which might be a consideration in time-sensitive
        applications.
5.  **Memory Utilization**:
    -   **Decision Tree** is the most memory-efficient model, with the
        lowest peak memory usage and memory increment.
    -   **KNN** and **Logistic Regression** have similar memory
        footprints, which are significantly higher than the Decision
        Tree but much lower than LDA.
    -   **LDA** requires the most memory, which could be a limiting
        factor in resource-constrained environments.

### Overall Conclusion:

-   **Efficiency and Performance Balance**: If you\'re looking for a
    balance between efficiency (both in terms of memory and training
    time) and performance (in terms of Precision-Recall AUC, ROC AUC,
    and F1 Score), **Logistic Regression** appears to be the best choice
    among the models evaluated.

-   **High Precision and Recall**: If the primary goal is to maximize
    precision and recall, and computational resources are less of a
    concern, **LDA** and **Logistic Regression** are strong contenders.

-   **Resource Constraints**: If memory usage and training time are
    critical constraints, **Decision Tree** offers a good balance,
    albeit with some trade-offs in terms of ROC AUC and F1 Score.

-   **Speed Priority**: For applications where training speed is the top
    priority, **KNN** stands out, though it does require a significant
    amount of memory and doesn\'t perform as well on ROC AUC.
:::

::: {.cell .markdown}
### Analysis of feature engineering methods

-   For Manual Feature Engineering
    -   Feature Hashing was considered but due to the large number of
        records (fearing collisions a lot) it was not implemented.
    -   Feature binning was done to FICO scores
    -   Created a new feature Debt to income Ratio by
        loan_amount/annual_inc
-   Applied Principal Component analysis but the accuracy remained
    somewhat unchanged hence not used in further evaluating
:::

::: {.cell .markdown}
After applying manual feature engineering and evaluating the models on
the test dataset, we observe the following results:

-   Logistic Regression Accuracy: 0.9106651218995653
-   KNN Accuracy: 0.875758368526616
-   LDA Accuracy: 0.8983069850880143
-   Decision Tree Accuracy: 0.8029425321722664

1.  **Logistic Regression** has shown a notable improvement with an
    accuracy of **91.07%**. This improvement indicates that the manual
    feature engineering steps were particularly effective for this
    model, likely due to the creation of features that better represent
    the underlying patterns in a way that logistic regression can
    leverage.

2.  **KNN** has an accuracy of **87.58%**. While this is a respectable
    score, the improvement from manual feature engineering might not be
    as pronounced as with Logistic Regression. This could be due to
    KNN\'s reliance on distance metrics, which might not benefit as much
    from the engineered features without further tuning of the distance
    metric or feature weighting.

3.  **LDA** shows an accuracy of **89.83%**, indicating a positive
    impact from the feature engineering. LDA benefits from features that
    help to linearly separate the classes, and the engineered features
    seem to aid in this aspect.

4.  **Decision Tree** has the lowest accuracy among the models at
    **80.29%**. While Decision Trees are inherently good at feature
    selection and can build complex decision boundaries, the manual
    features might not provide significant additional information or
    could even introduce complexity that doesn\'t improve the model\'s
    performance.

### Overall Conclusion: {#overall-conclusion}

-   **Most Effective Model**: After manual feature engineering,
    **Logistic Regression** stands out as the most effective model with
    the highest accuracy. This suggests that the transformations and new
    features introduced are particularly well-suited for a model that
    benefits from linear relationships.

-   **Moderate Improvements**: **LDA** also benefits from the feature
    engineering, showing a decent improvement. The engineered features
    likely help in defining a more separable linear space for LDA to
    operate in.

-   **Lesser Impact on KNN and Decision Tree**: **KNN** and **Decision
    Tree** show lesser improvements from manual feature engineering. KNN
    might require additional considerations such as feature scaling or
    distance metric tuning, and Decision Trees might already perform
    their form of feature engineering implicitly through splits.
:::

::: {.cell .markdown}
### Hyperparametric Tuning

-   Based on the Conclusion we have chosen Logistic Regression and LDA
    to be further optimized
:::

::: {.cell .markdown}
#### For Logistic Regression

-   **Parameter Grid**: Defined a grid of hyperparameters for Logistic
    Regression, focusing on the regularization strength (`C`) and the
    penalty type (`penalty`). A reduced range of `C` was used for faster
    processing and stronger regularization, and `l2` penalty was chosen
    for simplicity.
-   **GridSearchCV**: Utilized GridSearchCV to systematically work
    through the specified parameter grid, performing 5-fold
    cross-validation for each parameter combination, and identifying the
    combination that yielded the best accuracy.
-   **Parallel Processing**: Employed `n_jobs=-1` to use all available
    CPU cores, speeding up the grid search process.

### Model Evaluation:

-   **Best Parameters and Score**: After fitting GridSearchCV with the
    transformed training data, the best hyperparameters (`C=0.01`,
    `penalty='l2'`) and the best cross-validation accuracy score
    (`0.8696`) were printed.
-   **Validation Performance**: The Logistic Regression model, with the
    best hyperparameters, was evaluated on the validation data,
    resulting in an accuracy of `0.9153`.
-   **Test Performance**: Finally, the model was assessed on a separate
    test dataset, yielding an accuracy of `0.9150`, closely matching the
    validation performance.

### Conclusion for Logistic Regression:

The hyperparameter tuning process demonstrated the effectiveness of the
preprocessing steps combined with the optimized Logistic Regression
model. The selection of `C=0.01` indicates a preference for stronger
regularization, which likely helped prevent overfitting and contributed
to the model\'s robust generalization performance.

The model achieved high accuracy on both the validation and test
datasets (`~0.915`), underscoring its predictive capability. The close
agreement between validation and test accuracies suggests that the model
is well-calibrated and not overfitting to the training data.
:::

::: {.cell .markdown}
#### For LDA

### Phase 1: LDA with \'svd\' Solver (SVD stands for (Singular Value Decomposition))

-   **Solver**: You first conducted a grid search specifically for the
    \'svd\' solver, which does not support shrinkage. This solver is
    often effective for datasets where the number of features is large
    or when multicollinearity is present in the features.
-   **GridSearchCV**: A single-parameter grid specifying
    `{'solver': ['svd']}` was used to fit the LDA model using the
    \'svd\' solver.
-   **Fit**: The grid search was applied to the transformed training
    data (`X_train_transformed`), and the best parameters and scores
    were recorded.

### Phase 2: LDA with \'lsqr\' and \'eigen\' Solvers with Shrinkage

-   **Solver**: The second grid search targeted the \'lsqr\' and
    \'eigen\' solvers, both of which support shrinkage. Shrinkage can
    help improve model performance, especially when dealing with small
    sample sizes or highly correlated features.
-   **Shrinkage**: A range of shrinkage values, including \'auto\', was
    explored to determine the optimal level of regularization.
-   **GridSearchCV**: This grid search explored combinations of solvers
    and shrinkage values using the parameter grid defined in
    `param_grid_lda`.
-   **Error Handling**: The `error_score='raise'` parameter was set to
    raise errors immediately if any fit failed, providing immediate
    feedback on potential issues.

### Model Selection and Evaluation

-   **Best Model Selection**: After performing both grid searches, the
    code compares the best scores from each to select the overall best
    LDA model.
-   **Evaluation**: The best LDA model, determined to be the one with
    the \'svd\' solver, was then used to make predictions and evaluate
    performance on both validation and test datasets.

### Results and Conclusion:

-   **Best Parameters**: The grid search identified the \'svd\' solver
    as yielding the best cross-validation score (`0.8788`), indicating
    that, for your dataset, this solver without shrinkage was optimal.
-   **Validation Performance**: The LDA model with the \'svd\' solver
    achieved an accuracy of `0.8983` on the validation set, showcasing
    strong predictive performance.
-   **Test Performance**: Similarly, the model maintained its high
    accuracy on the test set (`0.8981`), suggesting good
    generalizability.
-   **Classification Report**: The detailed classification report for
    the test set revealed excellent precision and recall across both
    classes, with an overall accuracy of `0.90`. This indicates that the
    LDA model is highly effective in distinguishing between the two
    classes.
:::

::: {.cell .markdown}
### Comparing with the other models

The Logistic Regression model shows a significant improvement, achieving
the highest accuracy on the test set among all models. This suggests
that hyperparameter tuning and feature engineering had a substantial
positive impact. The LDA model also shows strong performance, indicating
that the chosen \'svd\' solver was effective for this dataset. The KNN
model\'s accuracy suggests an improvement, assuming the original model
had lower performance. The Decision Tree model\'s performance is the
lowest among the tuned models but might still represent an improvement
or be competitive with its original performance.

After tuning and applying manual feature engineering, the updated
performance metrics for the models on the test set are as follows:

-   **Logistic Regression Accuracy:** 0.9150128938221773 (Best
    parameters: {\'C\': 0.01, \'penalty\': \'l2\'})
-   **LDA Accuracy on Test Data:** 0.8981325758991416

The other two models

-   **Decision Tree Accuracy on Test Data:** 0.800712586114537
-   **KNN Accuracy on Test Data:** 0.8774401714193171

Comparing these results to the initial accuracies before tuning and
feature engineering:

-   **Logistic Regression:** Increased from 0.9106651218995653 to
    0.9150128938221773
-   **KNN:** Slight increase from 0.875758368526616 to
    0.8774401714193171
-   **LDA:** Decrease from initial 0.8983069850880143 (no change in
    accuracy after feature engineering)
-   **Decision Tree:** Decrease from initial 0.8029300743730613 to
    0.800712586114537

The Logistic Regression model saw a slight improvement after tuning and
feature engineering, indicating that these steps were beneficial for
this particular model. KNN also saw a minor increase in accuracy.
However, the LDA model did not show a change in accuracy, suggesting
that the manual feature engineering and tuning did not impact its
performance. The Decision Tree model experienced a slight decrease in
accuracy, which might indicate that the model became slightly overfitted
after the feature engineering process or that the changes made were not
beneficial for this model type.
:::
