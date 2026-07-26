import pandas as pd
from src.data_loader import HAIDataLoader
from src.preprocessing import HAIPreprocessor
from src.models import ClassicalModelsPipeline
from src.evaluation import evaluate_model, compare_models_summary

# 1. Chargement des données HAIEnd 23.05
print("1. Chargement des données...")
loader_train = HAIDataLoader("data/raw/end-train1.csv") # Baseline normale
df_train = loader_train.df

loader_test = HAIDataLoader("data/raw/end-test1.csv")   # Jeu de test avec attaques
df_test = loader_test.df

# 2. Prétraitement (Tabulaire 2D)
print("2. Prétraitement des données...")
preprocessor = HAIPreprocessor(scaler_type="minmax")

# Fit sur les données saines uniquement (fit_on_normal_only=True)
X_train_normal, _ = preprocessor.fit_transform_tabular(
    df_train, is_train=True, fit_on_normal_only=True
)
X_test, y_test = preprocessor.fit_transform_tabular(
    df_test, is_train=False
)

# 3. Entraînement des modèles classiques
print("3. Entraînement des modèles...")
pipeline = ClassicalModelsPipeline()

# Non supervisés : appris UNIQUEMENT sur X_train_normal
pipeline.train_unsupervised(X_train_normal)

# Supervisés : si tu souhaites entraîner XGBoost/RandomForest,
# utilise une portion de tes données étiquetées d'entraînement si disponible
# pipeline.train_supervised(X_train_sup, y_train_sup)

# 4. Inférence sur le jeu de test
print("4. Inférence sur le jeu de test...")
predictions = pipeline.predict_all(X_test)

# 5. Évaluation et Tableau Comparatif
print("5. Calcul des métriques...")
eval_results = {}
for model_name, out in predictions.items():
    eval_results[model_name] = evaluate_model(
        y_true=y_test, 
        y_pred=out["y_pred"], 
        model_name=model_name
    )

# Affichage du tableau comparatif final (exigé CDC Section 7.4)
summary_df = compare_models_summary(eval_results)
print("\n=== TABLEAU COMPARATIF DES MODÈLES DÉTECTION OT ===")
print(summary_df.to_string(index=False))

# 6. Sauvegarde des modèles pour le Dashboard Streamlit
pipeline.save_models("models_saved/")
preprocessor.save_scaler("models_saved/scaler.joblib")
print("\n✅ Modèles et scaler sauvegardés dans 'models_saved/'")