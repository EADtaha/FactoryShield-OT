import os
import pandas as pd
from sklearn.model_selection import train_test_split

# --- 1. CHEMINS ET DOSSIERS ---
BASE_DIR = "data/processed"
DIR_CLEAN = os.path.join(BASE_DIR, "clean_100")
DIR_MIXED = os.path.join(BASE_DIR, "mixed_70_30")

os.makedirs(DIR_CLEAN, exist_ok=True)
os.makedirs(DIR_MIXED, exist_ok=True)

print("--- 1. Chargement des données ---")
# Chargement des fichiers bruts/prétraités
df_train_clean = pd.read_csv(os.path.join(BASE_DIR, "train_clean_scaled.csv"))
df_test_features = pd.read_csv(os.path.join(BASE_DIR, "test_clean_scaled.csv"))
df_test_labels = pd.read_csv(os.path.join(BASE_DIR, "labels_clean.csv"))

# Normalisation des colonnes labels
df_test_full = df_test_features.copy()
df_test_full['label'] = df_test_labels.iloc[:, 0].values
df_train_clean['label'] = 0

print("--- 2. Extraction des Anomalies & Split 50/50 ---")
df_test_sain = df_test_full[df_test_full['label'] == 0].copy()
df_test_anom = df_test_full[df_test_full['label'] == 1].copy()

# Split continu (shuffle=False) des anomalies pour conserver les séquences physiques
anom_train, anom_test = train_test_split(
    df_test_anom, test_size=0.5, random_state=42, shuffle=False
)

n_anom_train = len(anom_train)
n_saines_req = int(n_anom_train * (70 / 30))

print("--- 3. Prélèvement d'un bloc continu de données saines ---")
# Prélèvement d'un bloc temporel continu (de la ligne 0 à n_saines_req)
saines_train = df_train_clean.iloc[0:n_saines_req].copy()

# Assemblage du Train Mixed 70/30
df_train_mixed_70_30 = pd.concat([saines_train, anom_train], axis=0).reset_index(drop=True)

# Assemblage du Test Set Final (Partagé par les deux dossiers)
df_test_final = pd.concat([df_test_sain, anom_test], axis=0).reset_index(drop=True)

print("--- 4. Organisation et sauvegarde des dossiers ---")

# 📁 DOSSIER 1 : 100% Clean
df_train_clean.to_csv(os.path.join(DIR_CLEAN, "train_clean_scaled.csv"), index=False)
df_test_final.to_csv(os.path.join(DIR_CLEAN, "test_set_final.csv"), index=False)

# 📁 DOSSIER 2 : Mixed 70/30
df_train_mixed_70_30.to_csv(os.path.join(DIR_MIXED, "train_mixed_70_30.csv"), index=False)
df_test_final.to_csv(os.path.join(DIR_MIXED, "test_set_final.csv"), index=False)

print("\n✅ Structure générée avec succès !")
print(f"📁 {DIR_CLEAN}/")
print(f"   ├── train_clean_scaled.csv  ({len(df_train_clean)} lignes, 100% Saines)")
print(f"   └── test_set_final.csv      ({len(df_test_final)} lignes)")
print(f"\n📁 {DIR_MIXED}/")
print(f"   ├── train_mixed_70_30.csv   ({len(df_train_mixed_70_30)} lignes, % anomalies = {df_train_mixed_70_30['label'].mean():.2%})")
print(f"   └── test_set_final.csv      ({len(df_test_final)} lignes)")
