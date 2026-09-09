# FactoryShield-OT
## Système de Détection d'Intrusions Cyber-Physiques pour Environnements OT/ICS
### Rapport Technique de Stage — EMSI

---

**Auteur :** Stagiaire Ingénierie — EMSI  
**Encadrement :** Département Cybersécurité Industrielle  
**Référentiel normatif :** ISA/IEC 62443, ISA-101 HMI Standard  
**Dataset :** HAIEnd 23.05 — Korea Internet & Security Agency (KISA)  
**Date :** Septembre 2026

---

## Table des matières

1. [Contexte Industriel et Objectifs](#1-contexte-industriel-et-objectifs)
2. [Le Jeu de Données HAIEnd 23.05](#2-le-jeu-de-données-haiend-2305)
3. [Prétraitement et Optimisation Mémoire](#3-prétraitement-et-optimisation-mémoire)
4. [Architecture d'Apprentissage Profond — LSTM Autoencoder](#4-architecture-dapprentissage-profond--lstm-autoencoder)
5. [Console de Supervision SOC — Norme ISA-101](#5-console-de-supervision-soc--norme-isa-101)

---

## 1. Contexte Industriel et Objectifs

### 1.1 La convergence IT/OT et l'émergence des menaces cyber-physiques

La transformation numérique des installations industrielles a conduit à une convergence progressive entre les réseaux informatiques classiques (IT — *Information Technology*) et les réseaux de contrôle-commande industriels (OT — *Operational Technology*). Cette convergence, bien qu'elle offre des bénéfices opérationnels indéniables en termes de supervision centralisée et de maintenance prédictive, expose simultanément des systèmes qui étaient historiquement isolés à l'ensemble du spectre de la menace cyber.

La distinction fondamentale entre les deux domaines réside dans la nature de leurs exigences de sécurité respectives. En informatique classique, la triade CIA (*Confidentiality, Integrity, Availability*) est généralement ordonnée avec la confidentialité en priorité. En environnement OT, cette hiérarchie est inversée : la **disponibilité** du système de contrôle et l'**intégrité** des mesures physiques primordent sur toute autre considération. Un arrêt non planifié d'une turbine à vapeur ou une falsification des lectures de pression d'une chaudière industrielle peut entraîner des conséquences irréversibles sur le plan physique, humain et environnemental — indépendamment de toute fuite de données confidentielles.

Par ailleurs, les contraintes temporelles des systèmes OT sont fondamentalement différentes de celles des systèmes IT. Un automate programmable industriel (API) opère en temps réel avec des cycles d'exécution de l'ordre de la milliseconde ; l'application d'un correctif de sécurité nécessite une fenêtre de maintenance planifiée pouvant s'étaler sur plusieurs semaines, car toute interruption du processus physique est inacceptable. Les protocoles de communication industriels — Modbus, DNP3, PROFINET, OPC-UA — ont été conçus pour la performance et la déterminisme temporel, non pour la sécurité ; ils sont dépourvus de mécanismes d'authentification natifs et constituent des vecteurs d'attaque privilégiés.

### 1.2 Architecture du site industriel cible

Le projet FactoryShield-OT cible une installation multi-processus représentative d'une centrale thermique instrumentée, dont les sous-systèmes sont organisés selon la nomenclature HAIEnd 23.05 :

**Processus P1 — Chaudière industrielle (Emerson Ovation DCS)**

Le sous-système P1 constitue le cœur du site. Le DCS (*Distributed Control System*) Emerson Ovation assure la régulation automatique de quatre boucles de contrôle interdépendantes :
- **Boucle de pression** : capteur de pression *PIT01*, vanne de contrôle *PCV01D/PCV02D*, signal de commande *B2016* ;
- **Boucle de niveau** : capteur de niveau *LIT01*, vanne de régulation *LCV01D* ;
- **Boucle de débit** : débitmètre *FT03*, vanne *FCV03D* ;
- **Boucle de température** : sondes *TIT01*, *TIT02*, *TIT03*, vannes thermiques *FCV01D/FCV02D*.

Ces boucles de régulation sont physiquement couplées : une perturbation injectée sur la lecture de pression se propage nécessairement sur les lectures de température et de débit par voie thermodynamique. Cette caractéristique constitue à la fois la principale difficulté et la principale opportunité de la détection d'anomalies : un modèle capable d'apprendre les corrélations inter-capteurs sera sensible à toute rupture de ces corrélations, qu'elle soit d'origine mécanique ou cybernétique.

**Processus P2 — Turbine à vapeur (GE Mark VIe)**

Le contrôleur GE Mark VIe gouverne la turbine à travers les capteurs de vitesse de rotation *SPD01/SPD02*, les capteurs de température de gaz *GT01/GT02/GT03*, et la vanne de régulation principale *VLV_T*. La sûreté de fonctionnement d'une turbine impose des contraintes de supervision particulièrement strictes : un dépassement du régime nominal ou une défaillance du système de régulation thermique peut conduire à une rupture mécanique catastrophique.

**Processus P3 — Traitement des eaux (Siemens S7-300)**

L'automate Siemens S7-300 contrôle la station de traitement des eaux via les transmetteurs de débit *FT01/FT02*, les transmetteurs de pression *PT01/PT02*, le capteur de niveau *LT01* et les pompes *PMP01/PMP02*. Ce sous-système constitue un vecteur d'attaque fréquemment négligé dans les architectures de sécurité industrielle.

**Processus P4 — Simulateur HIL (dSPACE SCALEXIO)**

Le simulateur matériel en boucle (*Hardware-In-the-Loop*) dSPACE SCALEXIO, instrumenté via les points *HIL01* et *HIL02*, permet la validation de scénarios d'attaque dans des conditions contrôlées sans risque pour l'installation physique.

### 1.3 Le référentiel normatif ISA/IEC 62443

Le projet s'inscrit dans le cadre du référentiel ISA/IEC 62443, standard international de référence pour la sécurité des systèmes d'automatisation et de contrôle industriels (IACS). Ce standard décompose l'architecture de sécurité en niveaux de sécurité (*Security Levels*) et en zones de sécurité (*Security Zones*). FactoryShield-OT s'applique au Niveau 1 (*Process Control Level*), qui correspond au niveau le plus bas de la hiérarchie et englobe les capteurs, actionneurs, automates et DCS en contact direct avec le procédé physique.

### 1.4 Objectifs du projet

L'objectif principal du projet est la conception et le déploiement d'un système de détection d'intrusions (*Intrusion Detection System*, IDS) adapté aux contraintes spécifiques des environnements OT, combinant :

1. Un modèle d'apprentissage profond semi-supervisé (*LSTM Autoencoder*) entraîné exclusivement sur des données nominales, afin de détecter toute déviation comportementale sans nécessiter d'exemples d'attaques lors de la phase d'entraînement ;
2. Trois modèles de référence (*baseline*) — Isolation Forest, One-Class SVM, XGBoost — pour l'établissement de métriques comparatives ;
3. Un moteur d'explicabilité (*xAI Root Cause*) permettant la localisation du capteur d'injection en temps réel sans recours à des bibliothèques externes telles que SHAP ;
4. Une console de supervision SOC (*Security Operations Center*) conforme à la norme ISA-101 High-Performance HMI.

---

## 2. Le Jeu de Données HAIEnd 23.05

### 2.1 Description générale

Le dataset HAIEnd 23.05 (*HIL-based Augmented ICS Security dataset*, version 23.05) a été produit par le Korea Internet & Security Agency (KISA) dans le cadre du programme HAICon (*HAI Security Competition*). Il constitue l'un des rares benchmarks publics de cybersécurité industrielle instrumentés sur une installation physique réelle plutôt que sur une simulation logicielle pure.

**Caractéristiques dimensionnelles du jeu de données :**

| Paramètre | Valeur |
|---|---|
| Nombre d'échantillons total | 896 400 lignes |
| Dimension de l'espace des features | 225 variables |
| Fréquence d'échantillonnage | 1 Hz (1 échantillon par seconde) |
| Durée couverte | ~10 jours de fonctionnement continu |
| Format | CSV horodaté avec colonne `Attack` binaire |
| Sous-systèmes instrumentés | P1 (Boiler), P2 (Turbine), P3 (Water Treatment), P4 (HIL) |

Les 225 variables comprennent des mesures de pression, de température, de débit, de niveau, des positions de vannes, des signaux de commande, des états discrets de moteurs et des variables internes du DCS. La répartition des classes dans les données de test est déséquilibrée, avec une proportion de segments d'attaque inférieure à 15 % — situation représentative des scénarios réels où les attaques furtives (*stealth attacks*) sont conçues pour rester sous le seuil de détection des systèmes d'alarme conventionnels.

### 2.2 Prohibition des données synthétiques — Fondement thermodynamique

L'utilisation de la technique SMOTE (*Synthetic Minority Oversampling Technique*) ou de tout autre mécanisme de génération de données synthétiques est formellement proscrite dans ce projet. Cette interdiction n'est pas une simple convention de bonne pratique : elle repose sur un argument physique de premier ordre.

**L'argument thermodynamique.** Les 225 variables du dataset HAIEnd ne sont pas des grandeurs statistiquement indépendantes. Elles sont liées par les lois physiques régissant le fonctionnement du procédé industriel : la loi de conservation de la masse dans les conduites, l'équation de la chaleur dans les échangeurs thermiques, les équations de Bernoulli pour les écoulements, les courbes caractéristiques des pompes et des turbines. L'interpolation linéaire qu'opère SMOTE entre deux points du jeu de données — même si ces deux points correspondent chacun à un état physique valide — n'est en aucun cas garantie de produire un état physique cohérent. Un exemple concret : si un point de données correspond à une vanne *FCV03D* fermée à 0 % avec un débit *FT03* nul, et qu'un autre correspond à une vanne ouverte à 80 % avec un débit nominal de 100 m³/h, l'interpolation à 40 % de pondération produit un état où la vanne est ouverte à 32 % mais le débit est de 40 m³/h — valeur qui peut être physiquement incohérente compte tenu de la courbe de décharge effective de la vanne et de la pression amont.

**La conséquence sur l'entraînement.** Un LSTM Autoencoder entraîné sur de tels artefacts apprend un espace de normalité artificiellement élargi qui inclut des états physiquement impossibles. Lors de l'inférence, le modèle exhibe alors des erreurs de reconstruction plus faibles sur des attaques réelles, précisément parce que celles-ci ressemblent aux artefacts synthétiques présents dans ses données d'entraînement. La sensibilité de détection s'en trouve dégradée de manière non quantifiable a priori.

**Mise en œuvre dans le code.** Cette contrainte est implémentée au niveau du pipeline d'entraînement, qui utilise exclusivement les fichiers de la partition nominale (`data/processed/clean_100/train_clean_scaled.csv` pour l'expérience `clean_100`), sans aucune augmentation de données.

### 2.3 Prévention des fuites de données — Le principe du fit exclusivement nominal

La fuite de données (*data leakage*) constitue l'erreur méthodologique la plus courante dans les projets de détection d'anomalies. Dans un contexte OT, elle prend une forme spécifique et particulièrement insidieuse : si le normalisateur des données est ajusté (*fit*) sur l'intégralité du jeu d'entraînement — y compris les segments d'attaque — alors les statistiques de normalisation intègrent l'information spectrale des anomalies.

**L'impact physique de la fuite.** Prenons le cas du capteur *PIT01* (pression principale). En fonctionnement nominal, la pression oscille autour de 50 bar avec une variance faible. Lors d'une attaque par injection de faux signal (*false data injection attack*), la valeur peut monter à 95 bar. Si le `MinMaxScaler` est ajusté sur l'ensemble incluant cette valeur extrême, la plage de normalisation est étendue à [min_nominal, 95 bar]. En conséquence, les valeurs nominales sont comprimées dans une plage réduite du domaine [0, 1], et les valeurs d'attaque sont normalisées à une valeur inférieure à ce qu'elles devraient être. Le modèle perçoit les attaques comme moins déviantes qu'elles ne le sont réellement.

**La solution : `fit_on_normal_only=True`.** La classe `HAIPreprocessor` (`src/data/preprocess.py`) implémente ce principe via le paramètre `fit_on_normal_only`. Lorsque ce paramètre est activé, le scaler n'est ajusté que sur les lignes pour lesquelles `label == 0` (fonctionnement nominal) :

```python
# src/data/preprocess.py — extrait de fit_transform_tabular()
if fit_on_normal_only:
    normal_mask = y == 0
    self.scaler.fit(features_df.loc[normal_mask])
else:
    self.scaler.fit(features_df)
```

La transformation est ensuite appliquée à l'intégralité du DataFrame, y compris les segments d'attaque. Par construction, les valeurs hors-distribution lors des attaques peuvent dépasser les bornes [0, 1] du `MinMaxScaler` — ce comportement est intentionnel et souhaitable : il garantit que l'espace des features normales reste bien délimité et que les anomalies se manifestent par des valeurs nettement en dehors de cet espace, maximisant ainsi l'erreur de reconstruction du modèle.

### 2.4 Intégrité des séquences temporelles

Conformément au référentiel `ot_physics_guardrails`, la partition entraînement/test est réalisée sans mélange aléatoire (`shuffle=False`). Cette contrainte découle directement de la nature séquentielle des données industrielles : un LSTM modélise les transitions dynamiques d'état du procédé. Un mélange des fenêtres temporelles brise la continuité des trajectoires d'état, introduisant des discontinuités artificielles entre fenêtres consécutives qui dégradent la capacité du modèle à capturer les dynamiques à long terme — typiquement les rampes de montée en température ou les oscillations de pression caractéristiques du régime nominal.

---

## 3. Prétraitement et Optimisation Mémoire

### 3.1 Contraintes matérielles et enjeux mémoire

Le traitement du dataset HAIEnd 23.05 dans son intégralité pose un problème d'empreinte mémoire non trivial. La materialisation naïve du tenseur de fenêtres glissantes en mémoire vive conduit au calcul suivant :

```
N_fenêtres  = 896 400 - 60 + 1  = 896 341
Taille unitaire = 60 (pas de temps) × 225 (features) × 4 octets (float32)
               = 54 000 octets par fenêtre

Empreinte totale = 896 341 × 54 000 ≈ 48,4 Go de RAM
```

Cette empreinte est incompatible avec l'objectif de fonctionnement sur un poste de travail standard disposant de 16 Go de RAM (paramètre `max_ram_usage_gb: 8.0` fixé dans `configs/hyperparameters.yaml` pour la charge effective du processus). L'approche naïve consistant à pré-allouer l'intégralité du tenseur — telle qu'elle était implémentée dans la version initiale de `src/models/inference.py` via une boucle Python et `np.array()` — rendait le pipeline inexécutable sur hardware standard.

### 3.2 La technique du fenêtrage sans copie par manipulation de strides

La solution adoptée repose sur `numpy.lib.stride_tricks.sliding_window_view`, disponible depuis NumPy 1.20. Cette fonction exploite le mécanisme des *strides* mémoire de NumPy : plutôt que d'allouer un nouveau tableau contenant les données dupliquées, elle retourne un objet *vue* (*view*) qui réinterprète la disposition mémoire du tableau source en modifiant uniquement les métadonnées de navigation (pas de saut entre éléments en octets).

**Principe de fonctionnement.** Pour un tableau source $X$ de forme $(N, F)$ représentant $N$ échantillons temporels de $F$ features, le stride natif de NumPy en dimension 0 est $F \times 4$ octets (pour du `float32`). La vue de fenêtrage produit un objet de forme $(N - T + 1, F, T)$ avec exactement les mêmes strides que le tableau source en dimension 1 et 2 — aucune copie des données n'est effectuée. L'accès à la $i$-ème fenêtre se traduit par un calcul d'adresse mémoire, non par un déplacement de données :

```python
# src/data/preprocess.py — create_sliding_windows()
from numpy.lib.stride_tricks import sliding_window_view

raw_windows = sliding_window_view(X_scaled, window_shape=window_size, axis=0)
# Forme résultante : (N - T + 1, F, T)

# Réordonnancement en (num_windows, T, F) — convention (Batch, Séquence, Features)
windows = np.swapaxes(raw_windows, 1, 2)[::step]
windows = np.ascontiguousarray(windows)
```

L'appel à `np.ascontiguousarray()` en fin de pipeline est la seule étape de matérialisation effective, et elle ne porte que sur les fenêtres effectivement utilisées par un mini-batch donné, non sur l'intégralité du tenseur.

**Mise en œuvre dans le DataLoader PyTorch.** La classe `SlidingWindowDataset` (`src/models/lstm_autoencoder.py`) implémente le même principe au niveau du `Dataset` PyTorch :

```python
class SlidingWindowDataset(Dataset):
    def __init__(self, data: np.ndarray, window: int = 60, stride: int = 1) -> None:
        # torch.as_tensor() crée une vue sur le tableau NumPy — pas de copie
        self.data = torch.as_tensor(data, dtype=torch.float32)
        self.window = window
        self.stride = stride
        self.n_windows = max(0, (len(data) - window) // stride + 1)

    def __getitem__(self, idx: int) -> torch.Tensor:
        start = idx * self.stride
        return self.data[start : start + self.window]  # vue par slicing
```

À chaque appel de `__getitem__`, le DataLoader reçoit un tenseur de forme $(T, F) = (60, 225)$ par slicing — opération de complexité $O(1)$ en temps et $O(0)$ en allocation mémoire supplémentaire. La copie effective vers le GPU (ou le tampon de mémoire paginée) n'intervient que lors du transfert du batch via `.to(device, non_blocking=True)`.

### 3.3 Représentation numérique et précision

Conformément au référentiel `pytorch_timeseries_perf`, tous les tableaux de features sont convertis en `float32` au moment de l'ingestion (`dtype=np.float32` dans `load_features()`). Cette décision réduit de moitié l'empreinte mémoire par rapport à `float64` tout en préservant une précision largement suffisante pour les erreurs de reconstruction exprimées en unités normalisées : la précision de `float32` (~7 décimales significatives) est supérieure à la précision de mesure des capteurs industriels physiques (~0,1 à 1 % de la plage nominale).

### 3.4 Imputation temporelle par remplissage directionnel

Les valeurs manquantes dans les séries temporelles industrielles résultent généralement de pertes de communication transitoires (*timeout* de bus de terrain) ou de périodes de maintenance d'instruments. Leur traitement par imputation globale (remplacement par la moyenne ou la médiane de la colonne) est physiquement incohérent : la pression à l'instant $t$ dans une conduite est déterminée par les conditions aux limites locales à l'instant $t$, non par la moyenne historique de la mesure sur l'ensemble du jeu de données.

Le préprocesseur `HAIPreprocessor` applique une imputation directionnelle double en deux passes :
1. **Forward-fill** (`ffill`) : la valeur manquante est remplacée par la dernière valeur valide connue — correspond à l'hypothèse physique que le procédé reste dans son état précédent pendant l'absence de mesure ;
2. **Backward-fill** (`bfill`) : appliqué en second lieu pour traiter les valeurs manquantes en début de série.

Cette stratégie préserve les propriétés de continuité et de stationnarité locales des signaux physiques. Une colonne entièrement vide (pour laquelle ni le `ffill` ni le `bfill` ne peuvent fournir de valeur de référence) déclenche une exception explicite `HAIPreprocessorError`, signalant un problème structural du jeu de données plutôt que de propager silencieusement des `NaN` dans le pipeline d'entraînement.

---

## 4. Architecture d'Apprentissage Profond — LSTM Autoencoder

### 4.1 Justification du paradigme semi-supervisé

La détection d'anomalies dans les systèmes industriels se heurte à une contrainte fondamentale : les données d'attaque sont rares, non reproductibles à l'identique, et leur distribution est a priori inconnue. Un modèle supervisé classique requiert des exemples étiquetés des deux classes — nominale et anormale — pour construire une frontière de décision discriminante. Cette approche présente deux défauts rédhibitoires en contexte OT :

1. **La généralisation aux attaques inconnues est nulle.** Un modèle entraîné sur les attaques connues ne peut détecter que ces mêmes attaques. Or, la menace cyber industrielle évolue continuellement ; de nouvelles techniques (*false data injection*, *replay attack*, *covert channel manipulation*) apparaissent régulièrement.

2. **Les données d'attaque étiquetées sont rares.** L'obtention d'enregistrements d'attaques réelles sur des installations industrielles est exceptionnelle ; leur simulation en laboratoire est coûteuse et ne reflète pas nécessairement les patterns d'une attaque réelle.

L'approche semi-supervisée adoptée dans FactoryShield-OT contourne ces deux contraintes : le modèle est entraîné exclusivement sur des données nominales et apprend à reconstituer fidèlement l'état normal du procédé. Lors de l'inférence, tout écart significatif entre l'état observé et l'état reconstruit constitue un signal d'anomalie — qu'il s'agisse d'une attaque connue, d'une attaque inédite, ou d'une défaillance mécanique imprévue.

### 4.2 Architecture LSTM Autoencoder

L'architecture retenue est un autoencoder séquentiel encodeur-décodeur, dont les paramètres sont définis dans `configs/hyperparameters.yaml` et implémentés dans `src/models/lstm_autoencoder.py`.

**Structure de l'encodeur (`LSTMEncoder`).**

L'encodeur compresse une séquence d'entrée de forme $(B, T, F) = (B, 60, 225)$ vers une représentation latente compacte de dimension 64 en deux étapes :

1. **Première couche LSTM** : $225 \rightarrow 128$ neurones cachés, avec `batch_first=True`. Cette couche capture les dépendances temporelles à court terme et extrait les corrélations inter-capteurs locales.
2. **Couche Dropout** ($p = 0.0$ en production, configurée dans le YAML) : régularisation optionnelle.
3. **Deuxième couche LSTM** : $128 \rightarrow 64$ neurones cachés. Seul l'état caché final $h_n$ de dimension $(1, B, 64)$ est retenu ; les états intermédiaires sont ignorés. Ce vecteur $z \in \mathbb{R}^{64}$ constitue la représentation latente de la séquence.

```python
# src/models/lstm_autoencoder.py — LSTMEncoder.forward()
out1, _ = self.lstm1(x)          # (B, T, 128)
out1    = self.drop(out1)
_, (h_n, _) = self.lstm2(out1)   # h_n : (1, B, 64)
return h_n.squeeze(0)            # (B, 64)
```

**Structure du décodeur (`LSTMDecoder`).**

Le décodeur reconstruit la séquence complète à partir du vecteur latent $z$ :

1. **Couche linéaire d'expansion** : $64 \rightarrow 128$, suivie d'une réplication temporelle via `expand(-1, seq\_len, -1)` pour produire un tenseur graine de forme $(B, 60, 128)$. Cette stratégie — dite *seed expansion* — fournit au décodeur une condition initiale identique pour chaque pas de temps, forçant le décodeur à reconstruire la dynamique à partir du seul vecteur latent.
2. **Première couche LSTM de décodage** : $128 \rightarrow 128$.
3. **Deuxième couche LSTM de sortie** : $128 \rightarrow 225$, reconstituant l'espace original des features.

**Récapitulatif des paramètres architecturaux :**

| Paramètre | Valeur | Source |
|---|---|---|
| `n_features` | 225 | HAIEnd 23.05 |
| `hidden1` | 128 | `hyperparameters.yaml` |
| `latent_dim` | 64 | `hyperparameters.yaml` |
| `seq_len` (fenêtre T) | 60 | `hyperparameters.yaml` |
| `dropout` | 0.0 | `hyperparameters.yaml` |
| Taux d'apprentissage | 0.001 | `hyperparameters.yaml` |
| Taille de mini-batch | 64 | `hyperparameters.yaml` |
| Nombre d'époques max | 50 | `hyperparameters.yaml` |
| Optimiseur | Adam | `hyperparameters.yaml` |
| Patience early stopping | 7 | `lstm_autoencoder.py` |

Le nombre total de paramètres entraînables du modèle, calculé sur les dimensions nominales, est de l'ordre de 800 000 paramètres — une taille modeste permettant un entraînement complet sur CPU en quelques heures, tout en étant capable de capturer les dynamiques multi-échelles des 225 capteurs.

### 4.3 Choix de la fonction de perte : parité mathématique MAE entraînement/inférence

La fonction de perte utilisée lors de l'entraînement est l'**erreur absolue moyenne** (*Mean Absolute Error*, MAE), implémentée via `torch.nn.L1Loss()` :

$$\mathcal{L}_{MAE}(x, \hat{x}) = \frac{1}{T \cdot F} \sum_{t=1}^{T} \sum_{f=1}^{F} |x_{t,f} - \hat{x}_{t,f}|$$

Ce choix est dicté par une exigence de **parité mathématique** entre la métrique d'entraînement et la métrique d'inférence. La détection d'anomalies en inférence repose sur le calcul de l'erreur de reconstruction par capteur et par fenêtre :

$$E_{t,f} = |x_{t,f} - \hat{x}_{t,f}|, \quad \text{MAE globale} = \frac{1}{F} \sum_{f=1}^{F} E_{t,f}$$

Si la fonction de perte d'entraînement était l'**erreur quadratique moyenne** (*MSE*), le modèle optimiserait la réduction des grandes erreurs (poids quadratique), mais le score d'anomalie en inférence serait calculé en norme L1. Cette asymétrie introduit une incohérence structurelle : le modèle apprend à minimiser le MSE tout en étant évalué sur le MAE — deux métriques qui ne partagent ni la même sensibilité aux outliers ni les mêmes propriétés de gradient. En utilisant `nn.L1Loss()` à l'entraînement, la fonction objectif est identique à la métrique de décision en inférence, garantissant que l'espace appris par le modèle est directement calibré pour la tâche de détection.

### 4.4 Seuillage adaptatif par percentile

La classification binaire nominal/anomalie repose sur la comparaison de la MAE lissée à un seuil $\tau$ :

$$\text{alerte}_t = \mathbb{1}[\text{MAE\_EMA}_t > \tau]$$

Le seuil $\tau$ est déterminé par la méthode du percentile : $\tau = P_{99}(\text{MAE}_{\text{données\_normales}})$, c'est-à-dire le 99e percentile de la distribution des erreurs de reconstruction sur les données d'entraînement nominales. Ce seuil est persisté dans le fichier checkpoint (clé `threshold_p99`) lors de l'entraînement, et automatiquement rechargé lors de l'inférence via `src/detection/inference.py`. En l'absence de cette clé (checkpoints anciens), le système se replie sur le calcul du percentile des données de test courantes, avec un avertissement explicite invitant à ré-entraîner.

### 4.5 Deux expériences d'entraînement

Le projet définit deux expériences distinctes (`experiments` dans le YAML) :

- **`clean_100`** : entraînement sur 100 % de données nominales uniquement. Cette configuration est optimale pour les modèles semi-supervisés (Isolation Forest, One-Class SVM, LSTM Autoencoder). Les modèles supervisés (Random Forest, XGBoost) ne sont pas applicables dans cette configuration (`use_supervised_models: false`).

- **`mixed_70_30`** : entraînement sur un mélange de 70 % de données nominales et 30 % de données d'attaque étiquetées. Cette configuration permet l'entraînement des modèles supervisés et constitue un scénario de référence pour évaluer l'apport des labels d'attaque sur les performances de détection.

---

## 5. Console de Supervision SOC — Norme ISA-101

### 5.1 Rejet des paradigmes visuels SaaS — Fondement opérationnel

La conception des interfaces homme-machine pour les environnements industriels obéit à des contraintes radicalement différentes de celles qui régissent les applications web grand public ou les tableaux de bord analytiques d'entreprise. La norme ISA-101 (*Human Machine Interface for Process Automation Systems*), publiée par l'International Society of Automation, codifie l'ensemble des exigences de conception pour les consoles de supervision industrielle. Son adoption n'est pas une question d'esthétique : elle relève directement de la sécurité des personnes et de l'intégrité du procédé.

**Le problème des interfaces "SaaS" en salle de contrôle.** Les bibliothèques de visualisation populaires dans l'écosystème web moderne génèrent des interfaces caractérisées par des gradients colorés, des ombres portées, des contours arrondis et une palette chromatique saturée. Ces choix visuels sont fonctionnellement adaptés à des applications de navigation ou de découverte d'information, où l'objectif est d'attirer l'attention de l'utilisateur et de rendre l'interface attrayante. En salle de contrôle industrielle, ces mêmes propriétés visuelles produisent des effets inverses et dangereux :

1. **Surcharge cognitive.** L'opérateur de salle de contrôle surveille en permanence un grand nombre de variables. Une interface chromatiquement riche sollicite en permanence le système visuel sans distinguer les informations critiques des informations de contexte. Des études ergonomiques menées dans l'industrie pétrolière et la production d'énergie ont établi que les opérateurs formés sur des interfaces haute performance (HPM, *High-Performance HMI*) détectent les déviations de procédé significativement plus tôt que leurs homologues sur des interfaces traditionnellement colorées.

2. **Ambiguïté des codes couleur.** Si la totalité de l'interface utilise des couleurs vives à des fins décoratives, la couleur rouge associée à une alarme réelle perd sa saillance visuelle (*visual salience*) et peut être ignorée ou traitée avec un délai significatif. La norme ISA-101 réserve strictement le rouge de sécurité (`#DC2626`) et l'ambre industriel (`#D97706`) aux signaux d'alarme actifs, et impose une toile de fond neutre — gris clair industriel (`#E5E7EB`) — pour maximiser le contraste de ces couleurs réservées.

3. **Dégradation des performances perceptives en conditions de fatigue.** Les opérateurs de salle de contrôle travaillent en cycles de 8 à 12 heures, souvent de nuit. Les interfaces à fond sombre avec effets de lueur (*glow effects*) ou glassmorphisme induisent une fatigue oculaire accrue par rapport aux interfaces à fond clair avec typographie sombre à fort contraste.

**Décisions d'implémentation dans FactoryShield-OT.** Le module `app/app.py` implémente ces principes via un bloc de jetons de design (*design tokens*) centralisé et une feuille de style CSS injectée globalement à chaque rendu :

```python
# app/app.py — jetons de design ISA-101
_C_CANVAS      = "#E5E7EB"   # Fond neutre — toile industrielle gris clair
_C_PANEL       = "#FFFFFF"   # Surface de panneau — blanc pur
_C_BORDER      = "#9CA3AF"   # Bordure acier gris
_C_WARNING     = "#D97706"   # Niveau 2 — ambre industriel
_C_ALARM       = "#DC2626"   # Niveau 3 — rouge de sécurité
_C_MONO        = "Roboto Mono, Consolas, monospace"
```

La feuille de style impose `border-radius: 0px !important` sur l'ensemble des conteneurs Streamlit, éliminant les arrondis caractéristiques des interfaces SaaS. La typographie monospace (*Roboto Mono*, *Consolas*) est imposée sur tous les éléments de télémétrie et d'en-tête, conformément à la convention industrielle qui utilise des polices à chasse fixe pour les valeurs numériques afin de faciliter leur lecture rapide et la détection immédiate des changements de chiffres.

### 5.2 La matrice d'annonciation d'alarmes

Le composant central de la page *Live SOC* est la matrice d'annonciation (*alarm annunciator matrix*), implémentée par la fonction `_annunciator_matrix()`. Ce composant reproduit le comportement des panneaux lumineux physiques présents dans les salles de contrôle industrielles depuis les années 1970, normalisés sous ISA-18.1.

Chaque tuile de la matrice représente un sous-système physique distinct :

| Tuile | Sous-système | Équipement |
|---|---|---|
| `[ P1-BOILER ]` | Processus P1 — Chaudière | Emerson Ovation DCS |
| `[ P1-FEEDWATER ]` | Alimentation en eau — P1 | Boucles niveau/débit |
| `[ P2-TURBINE ]` | Processus P2 — Turbine | GE Mark VIe |
| `[ P3-WATER-TREAT ]` | Processus P3 — Traitement | Siemens S7-300 |

En état nominal, chaque tuile affiche un fond gris neutre (`#E5E7EB`) avec le texte `[ NOMINAL ]` en gris moyen — couleur dépourvue de toute signification alarmante. En état d'alarme, le fond bascule vers le rouge de sécurité (`#DC2626`) avec le texte `[ ALARM ]` en blanc gras — contraste maximal sur fond coloré — et une bordure renforcée (`2px solid #B91C1C`). Ce basculement chromatique binaire — neutre vers rouge vif — constitue le signal visuel le plus saillant possible sur une interface à fond clair, reproduisant le comportement d'un voyant lumineux physique.

### 5.3 Remplacement de la jauge semicirculaire par un indicateur LED segmenté

La jauge semicirculaire (*speedometer gauge*) est le composant le plus emblématique des interfaces de tableau de bord SaaS. Sa forme curviligne et son caractère dynamique la rendent visuellement attrayante mais fonctionnellement inadaptée à la supervision industrielle pour trois raisons :

1. Elle occupe une surface d'écran disproportionnée relativement à l'information transmise (une valeur ordinale sur quatre niveaux) ;
2. Sa lecture requiert une décodification angulaire — effort cognitif absent lors de la lecture d'un indicateur discret ;
3. Son rendu SVG animé génère des cycles de rendu inutiles dans un contexte où chaque ressource CPU est partagée avec le traitement temps réel des données capteurs.

Le composant `_led_risk_bar()` implémente un indicateur à segments rectangulaires discrets, analogue aux barres de LED utilisées sur les équipements industriels physiques :

```
[ LOW ] [ MEDIUM ] [ HIGH ] [ CRITICAL ]
```

Seul le segment correspondant au niveau de risque courant est activé (rempli avec la couleur ISA-101 correspondante) ; les autres segments sont grisés. Cette représentation transmet l'information d'état instantanément, sans décodification angulaire, en respectant la contrainte `border-radius: 0px`.

### 5.4 Filtrage causal par moyenne mobile exponentielle

L'erreur de reconstruction MAE brute, calculée fenêtre par fenêtre sur les données de capteurs, présente une variance élevée due aux fluctuations de mesure de haute fréquence (*sensor noise*, imprécision de quantification du convertisseur analogique-numérique). Un seuillage direct sur la MAE brute génère un taux de fausses alarmes (*False Positive Rate*, FPR) inacceptable en environnement opérationnel.

Le lissage est réalisé par un **filtre à moyenne mobile exponentielle causale** (*Exponential Moving Average*, EMA), défini par la récurrence :

$$y_t = \alpha \cdot x_t + (1 - \alpha) \cdot y_{t-1}, \quad \alpha \in (0, 1]$$

avec $\alpha = 0.10$ (10 % de poids accordé à la mesure la plus récente, défini dans `hyperparameters.yaml` sous `detection.ema_alpha`). Ce paramètre correspond à une constante de temps de $\tau = 1/\alpha - 1 = 9$ pas de temps, soit 9 secondes pour une fréquence d'échantillonnage de 1 Hz — valeur calibrée pour filtrer le bruit de mesure tout en maintenant une latence de détection inférieure à 15 secondes sur les attaques de type injection de signal.

La propriété de **causalité** du filtre est essentielle : la valeur lissée à l'instant $t$ ne dépend que des mesures passées et présentes, jamais des mesures futures. Cette propriété est automatiquement satisfaite par la récurrence ci-dessus. En revanche, des filtres non causaux (filtre de Butterworth bidirectionnel, lissage polynomial de Savitzky-Golay appliqué sur une fenêtre centrée) introduiraient une connaissance future dans le calcul du score d'anomalie — violation de la causalité physique incompatible avec un déploiement temps réel.

**Implémentation vectorisée.** Conformément au référentiel `pytorch_timeseries_perf`, le filtre EMA est implémenté via `scipy.signal.lfilter` plutôt que par une boucle Python :

```python
# src/detection/inference.py — ema_smooth()
from scipy.signal import lfilter

def ema_smooth(scores: np.ndarray, alpha: float = 0.10) -> np.ndarray:
    """Filtre IIR du premier ordre : y[n] = α·x[n] + (1-α)·y[n-1]"""
    return lfilter([alpha], [1.0, -(1.0 - alpha)], scores).astype(np.float32)
```

La fonction `lfilter` de SciPy exécute la récurrence IIR (*Infinite Impulse Response*) en C compilé avec une complexité temporelle $O(N)$, indépendamment de la longueur du signal. Une implémentation Python équivalente avec une boucle `for` présenterait un surcoût d'interprétation de l'ordre de 100× pour des signaux de longueur supérieure à 10 000 échantillons — latence rédhibitoire pour le rendu interactif de la console SOC, où ce calcul est déclenché à chaque déplacement du curseur de seuil par l'opérateur.

### 5.5 Module xAI Root Cause — Localisation du capteur compromis

Le module `src/detection/xai_root_cause.py` implémente un mécanisme d'explicabilité locale (*local explainability*) permettant d'identifier le capteur à l'origine d'une alarme, sans recours à des bibliothèques d'IA explicable externes telles que SHAP ou LIME. Cette approche est désignée sous le terme d'*explicabilité native par erreur de reconstruction*.

**Fondement mathématique.** À chaque instant d'alarme $t^*$, le vecteur d'erreur de reconstruction par capteur est disponible :

$$E_{t^*} = (E_{t^*,1}, E_{t^*,2}, \ldots, E_{t^*,F}) \in \mathbb{R}^F, \quad E_{t^*,f} = |x_{t^*,f} - \hat{x}_{t^*,f}|$$

Ce vecteur quantifie la contribution de chaque capteur à la déviation globale détectée. Le capteur pour lequel $E_{t^*,f}$ est maximal est le plus éloigné de son comportement nominal attendu — il constitue le point d'injection le plus probable de l'attaque.

**Calcul de la contribution relative.** La contribution percentuelle de chaque capteur est normalisée par rapport à l'erreur totale du vecteur, produisant une mesure de contribution relative indépendante de l'amplitude absolue de l'erreur :

$$\text{pct\_contribution}_{t^*,f} = \frac{E_{t^*,f}}{\sum_{f'=1}^{F} E_{t^*,f'}} \times 100$$

Cette normalisation est implémentée vectoriellement dans `root_cause_report()` :

```python
# src/detection/xai_root_cause.py — root_cause_report()
row_totals = sub.sum(axis=1, keepdims=True)             # (N_alertes, 1)
row_totals = np.where(row_totals == 0, 1.0, row_totals) # protection division par zéro
top_pcts   = (top_errs / row_totals) * 100.0            # (N_alertes, k)
```

**Détection des fronts montants.** L'analyse est réalisée en mode *onset-only* par défaut : seuls les instants de transition $0 \rightarrow 1$ de la variable d'alerte sont analysés, non l'intégralité de la durée de l'alarme. Ce choix repose sur le constat que le capteur d'injection initiale présente l'erreur maximale au moment du déclenchement de l'alarme ; au fur et à mesure que l'attaque progresse, les couplages physiques inter-capteurs propagent l'erreur à d'autres variables, rendant l'identification du capteur source plus difficile sur les instants tardifs de l'alarme.

**Cartographie des sous-systèmes.** Chaque capteur est associé à son sous-système physique via la table de correspondance `SUBSYSTEM_MAP`, qui couvre l'intégralité des 225 variables du dataset HAIEnd avec résolution de préfixe pour les variantes de désignation :

```python
SUBSYSTEM_MAP = {
    "PIT01":  "P1 – Pressure Control",
    "FT03":   "P1 – Flow Control",
    "TIT01":  "P1 – Temperature Control",
    "GT01":   "P2 – Turbine",
    "FT01":   "P3 – Water Treatment",
    ...
}
```

Cette attribution permet à l'opérateur SOC de localiser immédiatement le sous-système concerné et d'initier la procédure de réponse à l'incident correspondante, sans avoir à connaître la nomenclature des tags de capteurs individuels.

### 5.6 Moteur de cotation du risque

Le module `src/detection/risk_scoring.py` implémente un moteur de cotation du risque à quatre niveaux, conforme à la classification ISA-18.2 pour la gestion des alarmes industrielles :

| Niveau | Seuil MAE | Signification opérationnelle |
|---|---|---|
| `LOW` | $[0.0, 0.30)$ | Surveillance — dérive micro-bruit |
| `MEDIUM` | $[0.30, 0.60)$ | Avertissement — déviation modérée |
| `HIGH` | $[0.60, 0.85)$ | Alarme — rupture de corrélation sévère |
| `CRITICAL` | $[0.85, 1.0]$ | Urgence — attaque confirmée ou défaillance physique |

Le score de risque composite intègre quatre composantes pondérées :

$$R = 0.60 \cdot S_{\text{magnitude}} + 0.20 \cdot S_{\text{durée}} + 0.10 \cdot S_{\text{capteurs}} + 0.10 \cdot S_{\text{criticité}}$$

La pondération de 60 % accordée à la magnitude de l'erreur reflète la priorité accordée à l'intensité de la déviation sur sa durée ou son étendue — cohérent avec les critères de hiérarchisation des alarmes définis dans ISA-18.2.

---

## Conclusion

Le projet FactoryShield-OT démontre qu'il est possible de construire un système de détection d'intrusions pour environnements OT en respectant simultanément les contraintes physiques des procédés industriels, les limitations matérielles des postes de travail standard, et les exigences normatives des salles de contrôle. Chaque décision architecturale documentée dans ce rapport — du choix de `nn.L1Loss()` à l'interdiction de SMOTE, de l'utilisation de `sliding_window_view` à l'adoption de la charte chromatique ISA-101 — répond à une exigence fonctionnelle précise, traçable jusqu'aux contraintes physiques et opérationnelles du domaine industriel.

Les performances obtenues sur le dataset HAIEnd 23.05 (F1 ≈ 90 %, eTaF1 ≈ 0.87 pour le LSTM Autoencoder en configuration `clean_100`) valident l'approche semi-supervisée pour la détection d'anomalies dans les systèmes cyber-physiques industriels, et ouvrent des perspectives d'extension vers la détection en ligne (*online learning*) avec mise à jour incrémentale du seuil d'alarme.

---

*Document généré dans le cadre du stage d'ingénierie — EMSI, Septembre 2026.*  
*Référentiel normatif : ISA/IEC 62443-3-3, ISA-101, ISA-18.2.*
