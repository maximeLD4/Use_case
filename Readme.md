# Maxime Le Doeuff : Data Scientist
## August 2023 Case Study
***
### Ce dossier contient plusieurs fichiers/dossiers
- 00_prepare_data.py
- 01_generate_model.py
- 02_generate_sequence.py
- 03_calculate_distribution.py
- 04_generate_match.py
- environment.yml
- Readme.md
- __match (JSON des match_1, match_2 et des match générés)__
- __models (Models enregistrés pour la génération de sequences de matchs)__
- __notebooks (Analyse des données de match_1 et match_2)__
- __out (Fichiers csv créés utiles à la génération de matchs)__
***
### Intégrer les données "match_1.json" et "matc_2.json" :
Pour la génération de match, le modèle est déja entrainé et sauvegardé, null besoins des deux JSON de match. Mais pour le reste, et les notebooks, il est necessaire de copier/coller les matchs 1 et 2 dans le repertoire "__match__".
***
### Installation du bon environnement :
Afin de ne rencontrer aucun problème de version, utiliser l'environnement adapté fournis. Pour cela, l'installer via la commande suivante:
```bash
conda env create -f environment.yml
```
***
### Détails des fichiers python
#### 00_prepare_data.py
Permet la création du dataframe utilisé pour entrainer le modèle de prédiction de séquences de jeu. 
A partir des matchs en fichier JSON. Génère le fichier seq_data_i.csv.

#### 01_generate_model.py
Creation et entrainement d'un modèle de "forecasting", exporté sous '_CNN_LSTM_Weights.keras_'.

#### 02_generate_sequence.py
Utilise le modèle '_CNN_LSTM_Weights.keras_' pour générer une sequence de jeu.

#### 03_calculate_distribution.py
Permet la création des dataframe '_norm_distribution.csv_' et '_size_distribution.csv_', qui régissent la distribution de la taille des vecteur de norme d'accéleration, ainsi que la distribution des accéleration pour chacun des labels.

#### 04_generate_match.py
Permet, à partir d'une séquence générée par '_generate_sequence.py_', de générer les normes associées aux labels. Ainsi on génére un match sous la forme attendue : '_match_created.json_'
***
### Analyse des données :
Tout ce qui concerne l'analyse des données et les réponses aux questions du pdf "... Data Scientist August 2023", se trouve dans "__notebooks__/analyses.ipynb"
***
## Génération de match

En executant le fichier '_f4_generate_match.py_' à l'aide de la commande suivante :
```bash
python3.10 f4_generate_match.py nb_minutes
```

Cela permet de générer un match de _'nb_minutes'_ minutes dans le dossier '___match___', avec le nom de fichier suivant : '_match_created_ _[___sequence_initiatrice___]_'

