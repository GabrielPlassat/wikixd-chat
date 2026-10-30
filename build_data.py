"""
Construit les données de l'app : récupère le wiki (export XML, ou API si elle est ouverte),
prépare les fiches et les passages, calcule les embeddings avec Albert,
enregistre tout dans le dossier data/.

Lancé depuis le notebook Colab.
"""
from __future__ import annotations

from datetime import datetime, timezone

import core


def construire(cle_albert: str, fichier_xml: str | None = None,
               modele_embeddings: str | None = None, max_pages: int | None = None,
               dossier=core.DATA_DIR) -> dict:
    client = core.client_albert(cle_albert)
    modeles = core.modeles_albert(client)
    print("Modèles Albert disponibles :", modeles)
    if not modele_embeddings:
        if not modeles["embeddings"]:
            raise RuntimeError("Aucun modèle d'embeddings trouvé sur votre compte Albert.")
        modele_embeddings = "bge-m3" if "bge-m3" in modeles["embeddings"] else modeles["embeddings"][0]
    print("→ modèle d'embeddings utilisé :", modele_embeddings)

    if fichier_xml:
        print(f"\n1-2/4 Lecture de l'export XML {fichier_xml}…")
        brutes = core.lire_export_xml(fichier_xml)
        print(f"   {len(brutes)} pages (articles, hors redirections)")
        if max_pages:
            brutes = brutes[:max_pages]
            print(f"   Mode test : limité à {len(brutes)} pages")
        source = f"export XML ({fichier_xml})"
    else:
        print("\n1/4 Liste des pages du wiki (API)…")
        titres = core.lister_pages(0)
        print(f"   {len(titres)} pages trouvées")
        if max_pages:
            titres = titres[:max_pages]
            print(f"   Mode test : limité à {len(titres)} pages")
        print("\n2/4 Récupération du contenu…")
        brutes = core.recuperer_pages(titres)
        source = core.WIKI_API

    print("\n3/4 Préparation des fiches et des passages…")
    pages = core.preparer_pages(brutes)
    passages = core.decouper(pages)
    print(f"   {len(pages)} pages, {len(passages)} passages")

    print("\n4/4 Calcul des embeddings avec Albert (peut prendre un moment)…")
    emb = core.calculer_embeddings(client, modele_embeddings, passages["chunk"].tolist())

    meta = core.sauvegarder(pages, passages, emb, {
        "date_aspiration": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "source": source,
        "modele_embeddings": modele_embeddings,
        "mode_test": bool(max_pages),
    }, dossier)
    print("\nTerminé :", meta)
    return meta
