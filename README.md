# Kayak NextStop

Trouvez votre prochaine destination selon **la météo** et les **hôtels**.

## Lien public
👉 https://chris-rattana-kayak-nextstop-kayak-nextstop-qn3boo.streamlit.app/

## Utilisation

1. Choisissez votre **profil voyageur** : Général, Plage, City Break, Randonnée, Évasion (tolérances météo préconfigurées).
2. Sélectionnez vos **dates** (Aller / Retour).
3. Réglez vos **paramètres** : **Pluie** (tolérance) et **Suggestions de villes** (quantité/affinage).
4. Parcourez le **Top 3**, puis la **carte** et le **tableau**.
5. Cliquez **Voir sur Kayak** pour explorer les hôtels.


## Données & API

- **Météo** : OpenWeatherMap (conditions actuelles + prévisions 24h et 5 jours).
- **Hôtels** : liens directs vers Kayak (pas d’accès aux prix ni aux notes car l’API officielle est fermée).
- **Cartographie** : OpenStreetMap (Overpass API pour afficher les hôtels à proximité).

## Déploiement local (optionnel)
```bash
pip install -r requirements.txt
streamlit run Kayak_NextStop.py
```

## Arborescence du projet
```
KKayak_NextStop/
├─ Kayak_NextStop.py
├─ requirements.txt
├─ assets/
│  ├─ Logo_Kayak_NextStop.png
│  └─ Favicon_Kayak_NextStop.png
├─ data/
│  └─ curated/
│     └─ destinations_scored.csv
└─ .streamlit/
   └─ config.toml
