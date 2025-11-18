import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import skops.io as sio

# 1. Chargement du jeu de données
drug_df = pd.read_csv("Data/heart_cleveland_upload.csv")
drug_df = drug_df.sample(frac=1, random_state=42)
print(drug_df.head(3))
# 2. Séparation Train / Test
X = drug_df.drop("condition", axis=1).values
y = drug_df.condition.values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=125
)

# 3. Pipelines d'apprentissage automatique
# Colonnes catégorielles : sex(1), cp(2), fbs(5), restecg(6), exang(8), slope(10), ca(11), thal(12)
# Colonnes numériques : age(0), trestbps(3), chol(4), thalach(7), oldpeak(9)
cat_col = [1, 2, 5, 6, 8, 10, 11, 12]
num_col = [0, 3, 4, 7, 9]

transform = ColumnTransformer(
    [
        ("num_imputer", SimpleImputer(strategy="median"), num_col),
        ("num_scaler", StandardScaler(), num_col),
        ("cat_imputer", SimpleImputer(strategy="most_frequent"), cat_col),
    ]
)

pipe = Pipeline(
    steps=[
        ("preprocessing", transform),
        ("model", RandomForestClassifier(n_estimators=100, random_state=125)),
    ]
)

pipe.fit(X_train, y_train)

# 4. Évaluation du modèle
predictions = pipe.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions, average="macro")

#  Utilisation de round(valeur, 2) au lieu de valeur.round(2)
print("Accuracy:", str(round(accuracy, 2) * 100) + "%", "F1:", round(f1, 2))

# Sauvegarde des métriques
with open("Results/metrics.txt", "w") as outfile:
    
    outfile.write(f"\nAccuracy = {round(accuracy, 2)}, F1 Score = {round(f1, 2)}.")

# Matrice de confusion
cm = confusion_matrix(y_test, predictions, labels=pipe.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=pipe.classes_)
disp.plot()
plt.savefig("Results/model_results.png", dpi=120)

# 5. Sauvegarde du modèle
sio.dump(pipe, "Model/heart_pipeline.skops")
