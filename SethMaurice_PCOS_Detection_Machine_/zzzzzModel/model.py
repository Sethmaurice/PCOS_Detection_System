import os
import matplotlib.pyplot as plt
import seaborn as sns

# ###ACCURACY########
# IMPORTS ALL importants LIBRARIES
import pandas as pds

# from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

# from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.impute import SimpleImputer

# Load your dataset
# df = pd.read_excel("PCOS_data.xlsx")
# Get the absolute path to the Excel file
df = pds.read_excel("PCOS_data.xlsx")

# Perform data cleaning (handle missing values, duplicates, outliers, etc.)
df_cleaned = df.dropna()  # Example: Remove rows with missing values
imputer = SimpleImputer(strategy="mean")


# Save the cleaned dataset
df_cleaned.to_csv("cleaned_dataset.csv", index=False)

# Remove duplicates
data_cleaned = df_cleaned.drop_duplicates()
# Summary statistics
summary_stats = data_cleaned.describe()

# Feature distributions
data_cleaned.hist(figsize=(15, 15))
# plt.show()

# Convert columns to numeric, handling errors and replacing invalid values
data_cleaned_numeric = data_cleaned.apply(pds.to_numeric, errors="coerce").fillna(0)

# Calculate correlation matrix
correlation_matrix = data_cleaned_numeric.corr()


# Correlation analysis
# correlation_matrix = data_cleaned.corr()


# # IMPORT DATASET
# MalariaData = pds.read_csv("malaria_clinical_data.csv")
# imputer = SimpleImputer(strategy="median")

X = df.drop(
    columns=[
        "Sl. No",
        "AMH(ng/mL)",
        "PCOS (Y/N)",
        "Patient File No.",
        "Weight (Kg)",
        "BMI",
        "Hb(g/dl)",
        "  I   beta-HCG(mIU/mL)",
        "II    beta-HCG(mIU/mL)",
        "FSH(mIU/mL)",
        "LH(mIU/mL)",
        "FSH/LH",
        "Waist:Hip Ratio",
    ]
)
y = df["PCOS (Y/N)"]
X_imputed = imputer.fit_transform(X)
# # columns_with_missing_values = ['temperature','wbc_count','rbc_count','hb_level','hematocrit','mean_cell_volume','mean_corp_hb', 'mean_cell_hb_conc', 'platelet_count', 'platelet_distr_width', 'mean_platelet_vl', 'neutrophils_count','lymphocytes_percent' ,'lymphocytes_count','mixed_cells_percent', 'mixed_cells_count', 'RBC_dist_width_Percent']

# # # Fill missing values with the mode of respective columns
# # for column in columns_with_missing_values:
# #     New_MalariaData = MalariaData[column].mode()[0]
# #     MalariaData[column].fillna(New_MalariaData, inplace=True)

# # print(MalariaData.info())
# # SPLIT(DIVISER) DATASET INTO TRAINING SET AND TEST SET

X_for_train, x_for_test, y_for_train, y_for_test = train_test_split(
    X_imputed, y, test_size=0.2
)
# Split the dataset into training and testing sets
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# # Create a Decision Tree, Logistic Regression, Support Vector Machine and Random Forest Classifiers
Decision_Tree_Model = DecisionTreeClassifier()
# Logistic_Regression_Model = LogisticRegression()
Logistic_Regression_Model = LogisticRegression(max_iter=1000)
# Support_Vector_Machine_Model = svm.SVC(kernel="linear")
Random_Forest_Model = RandomForestClassifier(n_estimators=100)


# # TRAIN THE MODEL USING THE TRAINING SETS
Decision_Tree_Model.fit(X_for_train, y_for_train)
Logistic_Regression_Model.fit(X_for_train, y_for_train)
# Support_Vector_Machine_Model.fit(X_for_train, y_for_train)
Random_Forest_Model.fit(X_for_train, y_for_train)

# # PREDICT THE MODEL
DT_prediction = Decision_Tree_Model.predict(x_for_test)
LR_prediction = Logistic_Regression_Model.predict(x_for_test)
# SVM_prediction = Support_Vector_Machine_Model.predict(x_for_test)
RF_prediction = Random_Forest_Model.predict(x_for_test)

# # CALCULATION OF MODEL ACUURACY
DT_score = accuracy_score(y_for_test, DT_prediction)
LR_score = accuracy_score(y_for_test, LR_prediction)
# SVM_score = accuracy_score(y_for_test, SVM_prediction)
RF_score = accuracy_score(y_for_test, RF_prediction)

# # DISPLAY ACCURACY
print("Decistion Tree accuracy =", DT_score * 100, "%")
print("Logistic Regression accuracy =", LR_score * 100, "%")
# print("Suport Vector Machine accuracy =", SVM_score * 100, "%")
print("Random Forest accuracy =", RF_score * 100, "%")
# ##After accuracy testing (except SVM) the most accurate is Random Forest Model
# #######################################

import pandas as pds
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.impute import SimpleImputer

# MalariaData = pds.read_csv("malaria_clinical_data.csv")
# df = pd.read_excel("PCOS_data.xlsx")
file_path = os.path.abspath("PCOS_data.xlsx")

# Load the Excel dataset
df = pds.read_excel("PCOS_data.xlsx")

df_cleaned = df.dropna()  # Example: Remove rows with missing values
##Replace missing values with the mode of the column

X = df.drop(
    columns=[
        "Sl. No",
        "AMH(ng/mL)",
        "PCOS (Y/N)",
        "Patient File No.",
        "Weight (Kg)",
        "BMI",
        "Hb(g/dl)",
        "  I   beta-HCG(mIU/mL)",
        "II    beta-HCG(mIU/mL)",
        "FSH(mIU/mL)",
        "LH(mIU/mL)",
        "FSH/FSH/LH",
        "Waist:Hip Ratio",
    ]
)


y = df["PCOS (Y/N)"]
# X_imputed = imputer.fit_transform(X)
X_imputed = df_cleaned.fit_transform(X)

model = RandomForestClassifier(n_estimators=100)

model.fit(X_imputed, y)

# TO SAVE THE TRAINED MODEL#
joblib.dump(model, "sethonde.joblib")
