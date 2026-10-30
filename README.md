# Discuter avec le wiki de l'eXtrême Défi

Prototype d'assistant conversationnel qui répond aux questions sur le [wiki de la Fabrique des Mobilités / XD](https://wikixd.fabmob.io), avec l'API Albert (IA souveraine de l'État).

## Comment ça marche

Chaque question est d'abord « routée » par Albert vers la bonne source :

| Type de question | Source utilisée | Exemple |
|---|---|---|
| Lister, compter, répartir | **Table des fiches** : une ligne par page, avec ses catégories et tous les champs de sa fiche (Thème, Tags, Acteur…) | « Combien de véhicules Proto 2024 ? » |
| Question ouverte | **Recherche dans le texte** des pages (recherche sémantique) | « Comment financer un projet de véli ? » |
| Les deux | Table + texte | « Quels acteurs font de la réparabilité, et comment ? » |

Les comptages sont faits par le programme, pas par le modèle : Albert ne fait que rédiger à partir des résultats, ce qui évite les chiffres inventés. Chaque réponse affiche ses sources cliquables.

Les champs viennent directement du texte source des fiches, ce qui est plus complet que les propriétés sémantiques du wiki (beaucoup ne sont pas déclarées).

## Contenu du dépôt

- `construire_donnees.ipynb` : notebook Colab qui aspire le wiki et construit l'index (à relancer pour rafraîchir)
- `app.py` : l'application Streamlit
- `core.py` : la logique commune (wiki, Albert, recherche)
- `build_data.py` : l'aspiration, appelée par le notebook
- `data/` : créé par le notebook (pages, passages, index)

## Mise en route (sans rien installer sur votre ordinateur)

1. **Créer le dépôt GitHub** : nouveau dépôt `wikixd-chat` (public, ou privé avec Streamlit connecté à votre compte), puis *Add file → Upload files* et déposer tous les fichiers.
2. **Construire les données dans Colab** : ouvrir `construire_donnees.ipynb` dans Colab (*Fichier → Ouvrir → GitHub*), ajouter le secret `ALBERT_API_KEY`, remplacer l'adresse du dépôt dans la 1re cellule, puis lancer les cellules dans l'ordre. Commencer par le **test sur 30 pages**.
3. **Mettre les données dans le dépôt** : envoi automatique (secret `GITHUB_TOKEN`) ou téléchargement du zip puis dépôt dans un dossier `data/` sur GitHub.
4. **Déployer sur Streamlit Cloud** : *New app* → dépôt `wikixd-chat`, fichier `app.py`. Dans *Advanced settings → Secrets* :
   ```
   ALBERT_API_KEY = "votre-clé"
   ```

## Rafraîchir les données

Relancer la section 3 et 4 du notebook. L'app affiche la date d'aspiration dans la barre latérale.

## Limites connues

- Photo du wiki à une date donnée (pas de lecture en temps réel).
- La qualité du routage dépend du modèle Albert choisi (préférer le plus grand disponible).
- Les filtres « contient » sont simples : « réparabilité » trouve « Réparabilité », mais pas les synonymes.
- Le contenu du wiki est sous licence CC BY-SA : citer les pages en cas de réutilisation.
