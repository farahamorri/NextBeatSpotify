# NextBeatSpotify



C'est un choix **parfait** pour tes contraintes. Le dataset Spotify est constitué de données tabulaires (chiffres et texte dans un CSV), ce qui est extrêmement **léger** à traiter. Ton ordinateur ne va même pas chauffer.

Voici le projet idéal qui combine **Ressources limitées + TP Autoencoders + Application Cool**.

-----

### Titre du Projet : Le "Spotify Smart Recommender"

**Le concept :** Construire un moteur de recommandation musicale. L'utilisateur entre une chanson qu'il aime, et ton modèle lui propose 5 chansons similaires basées sur l'audio (danceability, energy, acousticness...) et non sur le genre déclaré.

**Lien avec le TP :** **Autoencoders**.
Dans les TPs, tu as probablement utilisé l'autoencodeur pour compresser ou débruiter des images. Ici, tu vas l'utiliser pour **comprendre l'essence mathématique d'une chanson**.

-----

### Pourquoi ça répond aux consignes "Facile & Ressources Limitées" ?

1.  **Données minuscules :** Un fichier CSV de 100 000 lignes pèse quelques mégaoctets (contre des gigaoctets pour des images).
2.  **Entraînement éclair :** Un Autoencoder simple (Dense Layers) sur ces données s'entraîne en **moins de 2 minutes** sur un CPU standard.
3.  **Contribution significative :** Le TP classique fait de la reconstruction. Toi, tu vas utiliser **l'espace latent** (le goulot d'étranglement au milieu du modèle) pour calculer la similarité entre les musiques. C'est une utilisation avancée mais facile à coder.

-----

### La Feuille de Route Technique (Step-by-Step)

Voici comment tu vas structurer ton projet pour avoir une super note sans galérer :

#### 1\. Préparation des données (Data Prep)

Tu gardes uniquement les colonnes numériques (`valence`, `acousticness`, `danceability`, `energy`, `loudness`, `tempo`, etc.).

  * **Important :** Normalise les données (Mise à l'échelle entre 0 et 1 avec `MinMaxScaler`). Les Autoencodeurs détestent les données non normalisées.

#### 2\. L'Architecture (Le Modèle)

Tu construis un Autoencodeur simple avec Keras ou PyTorch.

  * **Entrée :** Nombre de features audio (ex: 12 colonnes).
  * **Encoder :** Compresse l'info (ex: 12 -\> 8 -\> **3 neurones**).
  * **Bottleneck (Espace Latent) :** Ces **3 neurones** représentent "l'ADN" de la chanson.
  * **Decoder :** Essaie de reconstruire les données (3 -\> 8 -\> 12).

#### 3\. L'Entraînement

Tu entraînes le modèle pour qu'il apprenne à reproduire l'entrée en sortie.

  * *Loss function :* MSE (Mean Squared Error).
  * *Epochs :* 10 à 20 suffisent largement.

#### 4\. La "Contribution Significative" (La partie intelligente)

Une fois entraîné, tu te fiches de la sortie. Ce qui t'intéresse, c'est le **milieu** (l'Encoder).

  * Tu coupes le modèle en deux.
  * Tu passes toutes tes chansons dans l'Encoder pour obtenir leur version compressée (vecteur de 3 chiffres).
  * Tu utilises un algorithme simple (KNN ou Cosine Similarity de Scikit-Learn) pour trouver les vecteurs les plus proches.

#### 5\. L'Application Concrète (Interface)

Avec **Streamlit**, tu fais une interface simple :

1.  Une liste déroulante pour choisir une chanson (ex: "Shape of You").
2.  Ton code cherche les voisins les plus proches dans l'espace latent.
3.  Il affiche : *"Si tu aimes Shape of You, tu aimeras ces 5 titres..."*

-----

### Exemple de code pour te lancer (Squelette)

Voici la partie "Modèle" pour te montrer à quel point c'est léger :

```python
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

# 1. Load Data (Supposons que df est ton dataset nettoyé)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(df[['danceability', 'energy', 'valence', 'tempo', 'acousticness']])

# 2. Architecture Autoencoder (Très léger)
input_dim = X_train.shape[1] # Nombre de colonnes
encoding_dim = 3 # On compresse à 3 dimensions pour visualiser facilement si besoin

input_layer = Input(shape=(input_dim,))
encoded = Dense(8, activation='relu')(input_layer)
bottleneck = Dense(encoding_dim, activation='relu')(encoded) # L'ADN de la musique
decoded = Dense(8, activation='relu')(bottleneck)
output_layer = Dense(input_dim, activation='sigmoid')(decoded)

autoencoder = Model(input_layer, output_layer)
encoder = Model(input_layer, bottleneck) # On garde ça pour la recommandation

# 3. Compile & Train
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(X_train, X_train, epochs=20, batch_size=32, shuffle=True)

# 4. Utilisation pour recommandation
compressed_songs = encoder.predict(X_train)
# Ensuite, tu utilises NearestNeighbors de sklearn sur 'compressed_songs'
```

C'est un projet très propre, scientifiquement valide (réduction de dimension), et qui tourne sur n'importe quel vieux PC.

**Est-ce que ça te convient ? Je peux te donner le code de la partie "Recommandation (Nearest Neighbors)" si tu veux.**