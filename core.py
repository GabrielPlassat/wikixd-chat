"""
Module commun : aspiration du wiki XD (MediaWiki + Semantic MediaWiki),
préparation des données, appels à l'API Albert, recherche hybride.

Utilisé par le notebook Colab (construction des données) et par l'app Streamlit.
"""
from __future__ import annotations

import json
import re
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import requests

WIKI_API = "https://wikixd.fabmob.io/api.php"
WIKI_PAGE_URL = "https://wikixd.fabmob.io/wiki/"
ALBERT_BASE_URL = "https://albert.api.etalab.gouv.fr/v1"
USER_AGENT = "WikiXD-Chat-POC/0.1 (prototype non commercial, lecture seule)"
DATA_DIR = Path(__file__).parent / "data"

# Taille max d'un fichier d'embeddings (le téléversement web GitHub est limité à 25 Mo)
SHARD_MAX_BYTES = 20 * 1024 * 1024


# ---------------------------------------------------------------------------
# Outils texte
# ---------------------------------------------------------------------------
def normaliser(texte: str) -> str:
    """Minuscules, sans accents : pour comparer « Véhicule » et « vehicule »."""
    texte = unicodedata.normalize("NFKD", str(texte or ""))
    return "".join(c for c in texte if not unicodedata.combining(c)).lower().strip()


def url_page(titre: str) -> str:
    return WIKI_PAGE_URL + requests.utils.quote(titre.replace(" ", "_"), safe="/:'(),")


# ---------------------------------------------------------------------------
# 1. Aspiration du wiki (API MediaWiki standard)
# ---------------------------------------------------------------------------
def appel_wiki(params: dict, pause: float = 0.5, essais: int = 4) -> dict:
    params = {**params, "format": "json", "formatversion": "2"}
    for essai in range(essais):
        try:
            r = requests.get(WIKI_API, params=params, timeout=60,
                             headers={"User-Agent": USER_AGENT})
            if r.status_code in (401, 402, 403):
                raise PermissionError(
                    f"Le wiki refuse l'accès automatisé (code {r.status_code}, protection anti-robots). "
                    "Utilisez un export XML du wiki : voir la section « Export » du notebook.")
            r.raise_for_status()
            time.sleep(pause)  # politesse envers le serveur
            return r.json()
        except PermissionError:
            raise
        except Exception as e:  # noqa: BLE001
            if essai == essais - 1:
                raise
            print(f"  … nouvel essai ({e})")
            time.sleep(3 * (essai + 1))
    return {}


def statistiques_wiki() -> dict:
    return appel_wiki({"action": "query", "meta": "siteinfo",
                       "siprop": "statistics"})["query"]["statistics"]


def lister_pages(espace: int = 0) -> list[str]:
    """Tous les titres d'un espace de noms (0 = articles, hors redirections)."""
    titres, suite = [], {}
    while True:
        data = appel_wiki({"action": "query", "list": "allpages", "apnamespace": espace,
                           "aplimit": "max", "apfilterredir": "nonredirects", **suite})
        titres += [p["title"] for p in data["query"]["allpages"]]
        if "continue" not in data:
            return titres
        suite = data["continue"]


def recuperer_pages(titres: list[str], lot: int = 50) -> list[dict]:
    """Texte source (wikitext), catégories et date de chaque page, par lots de 50."""
    pages: dict[str, dict] = {}
    for i in range(0, len(titres), lot):
        paquet = titres[i:i + lot]
        suite: dict = {}
        while True:
            data = appel_wiki({"action": "query", "titles": "|".join(paquet),
                               "prop": "revisions|categories",
                               "rvprop": "content|timestamp", "rvslots": "main",
                               "cllimit": "max", "clshow": "!hidden", **suite})
            for p in data.get("query", {}).get("pages", []):
                if p.get("missing"):
                    continue
                fiche = pages.setdefault(p["title"], {"title": p["title"], "categories": [],
                                                      "wikitext": "", "timestamp": ""})
                if p.get("revisions"):
                    rev = p["revisions"][0]
                    fiche["wikitext"] = rev.get("slots", {}).get("main", {}).get("content", "")
                    fiche["timestamp"] = rev.get("timestamp", "")
                for c in p.get("categories", []):
                    nom = c["title"].split(":", 1)[-1]
                    if nom not in fiche["categories"]:
                        fiche["categories"].append(nom)
            if "continue" not in data:
                break
            suite = data["continue"]
        print(f"  {min(i + lot, len(titres))}/{len(titres)} pages récupérées")
    return list(pages.values())


def lire_export_xml(chemin: str | Path, espace: int = 0) -> list[dict]:
    """
    Lit un export MediaWiki (fichier XML produit par Spécial:Exporter ou par
    les administrateurs avec dumpBackup.php). Garde la dernière version de chaque page.
    Les catégories viennent des liens [[Catégorie:…]] et des noms de modèles de fiche
    (une fiche {{Acteur|…}} range la page dans la catégorie Acteur).
    """
    import xml.etree.ElementTree as ET

    import mwparserfromhell

    def nom(balise):
        return balise.rsplit("}", 1)[-1]

    pages: dict[str, dict] = {}
    for _, el in ET.iterparse(str(chemin), events=("end",)):
        if nom(el.tag) != "page":
            continue
        champs = {nom(c.tag): c for c in el}
        ns = int((champs.get("ns").text if champs.get("ns") is not None else "0") or 0)
        if ns != espace or champs.get("redirect") is not None:
            el.clear()
            continue
        titre = champs["title"].text
        revs = [c for c in el if nom(c.tag) == "revision"]
        if revs:
            rev = {nom(c.tag): c for c in revs[-1]}
            texte = rev["text"].text if rev.get("text") is not None else ""
            date = rev["timestamp"].text if rev.get("timestamp") is not None else ""
            texte = texte or ""
            cats = re.findall(r"\[\[\s*(?:Cat[ée]gorie|Category)\s*:\s*([^\]|]+)", texte, re.I)
            try:
                modeles = [str(m.name).strip() for m in
                           mwparserfromhell.parse(texte).filter_templates(recursive=False)
                           if any(str(p.name).strip() and not str(p.name).strip().isdigit()
                                  for p in m.params)]
            except Exception:  # noqa: BLE001
                modeles = []
            categories = []
            for c in [c.strip() for c in cats] + modeles:
                if c and c not in categories:
                    categories.append(c)
            pages[titre] = {"title": titre, "wikitext": texte,
                            "timestamp": date, "categories": categories}
        el.clear()
    return list(pages.values())


# ---------------------------------------------------------------------------
# 2. Transformation des pages : champs des fiches + texte lisible
# ---------------------------------------------------------------------------
def analyser_wikitext(wikitext: str) -> tuple[dict, str]:
    """
    Renvoie (champs, texte).
    - champs : paramètres des modèles (fiches Acteur, Projet, Véhicule…), ex. {"Theme": "..."}
      C'est plus complet que les propriétés sémantiques, dont beaucoup ne sont pas déclarées.
    - texte : contenu lisible (texte de la page + champs des fiches en « Champ : valeur »).
    """
    import mwparserfromhell

    code = mwparserfromhell.parse(wikitext or "")
    champs: dict[str, str] = {}
    lignes_champs: list[str] = []
    for modele in code.filter_templates(recursive=False):
        for param in modele.params:
            nom = str(param.name).strip()
            if not nom or nom.isdigit():
                continue
            valeur = mwparserfromhell.parse(str(param.value)).strip_code().strip()
            valeur = re.sub(r"\s+", " ", valeur)
            if not valeur:
                continue
            cle = nom if nom not in champs else f"{nom}_{modele.name.strip()}"
            champs[cle] = valeur
            lignes_champs.append(f"{nom} : {valeur}")

    corps = code.strip_code(normalize=True, collapse=True)
    corps = re.sub(r"\n{3,}", "\n\n", corps).strip()
    texte = "\n".join(lignes_champs)
    if corps:
        texte = (texte + "\n\n" + corps).strip()
    return champs, texte


def preparer_pages(pages_brutes: list[dict]) -> pd.DataFrame:
    lignes = []
    for p in pages_brutes:
        champs, texte = analyser_wikitext(p.get("wikitext", ""))
        lignes.append({
            "title": p["title"],
            "url": url_page(p["title"]),
            "categories": p.get("categories", []),
            "fields": champs,
            "text": texte,
            "timestamp": p.get("timestamp", ""),
        })
    return pd.DataFrame(lignes)


def decouper(df_pages: pd.DataFrame, taille: int = 1500, chevauchement: int = 200,
             min_caracteres: int = 80) -> pd.DataFrame:
    """Découpe chaque page en passages, chacun préfixé par le titre et les catégories."""
    passages = []
    for _, p in df_pages.iterrows():
        texte = (p["text"] or "").strip()
        if len(texte) < min_caracteres:
            continue
        entete = f"Page : {p['title']}\n"
        if len(p["categories"]):
            entete += "Catégories : " + ", ".join(p["categories"]) + "\n"
        debut = 0
        while debut < len(texte):
            fin = min(debut + taille, len(texte))
            if fin < len(texte):  # couper proprement sur une fin de paragraphe ou de phrase
                coupe = max(texte.rfind("\n", debut + taille // 2, fin),
                            texte.rfind(". ", debut + taille // 2, fin))
                if coupe > debut:
                    fin = coupe + 1
            passages.append({"title": p["title"], "url": p["url"],
                             "chunk": entete + texte[debut:fin].strip()})
            if fin >= len(texte):
                break
            debut = max(fin - chevauchement, debut + 1)
    return pd.DataFrame(passages)


# ---------------------------------------------------------------------------
# 3. API Albert (compatible OpenAI)
# ---------------------------------------------------------------------------
def client_albert(cle_api: str):
    from openai import OpenAI
    return OpenAI(base_url=ALBERT_BASE_URL, api_key=cle_api, timeout=120, max_retries=3)


def modeles_albert(client) -> dict[str, list[str]]:
    """{"chat": [...], "embeddings": [...]} d'après GET /v1/models."""
    res = {"chat": [], "embeddings": []}
    for m in client.models.list().data:
        type_ = getattr(m, "type", "") or (m.model_extra or {}).get("type", "")
        if type_ == "text-generation":
            res["chat"].append(m.id)
        elif type_ == "text-embeddings-inference":
            res["embeddings"].append(m.id)
    return res


def calculer_embeddings(client, modele: str, textes: list[str], lot: int = 32,
                        pause: float = 0.2) -> np.ndarray:
    vecteurs = []
    for i in range(0, len(textes), lot):
        paquet = textes[i:i + lot]
        for essai in range(6):
            try:
                r = client.embeddings.create(model=modele, input=paquet)
                vecteurs += [d.embedding for d in sorted(r.data, key=lambda d: d.index)]
                break
            except Exception as e:  # noqa: BLE001  (quota dépassé, coupure réseau…)
                if essai == 5:
                    raise
                attente = 5 * (essai + 1)
                print(f"  … pause {attente}s ({type(e).__name__})")
                time.sleep(attente)
        if (i // lot) % 20 == 0:
            print(f"  {min(i + lot, len(textes))}/{len(textes)} passages vectorisés")
        time.sleep(pause)
    m = np.asarray(vecteurs, dtype=np.float32)
    m /= np.linalg.norm(m, axis=1, keepdims=True) + 1e-12
    return m


def discuter(client, modele: str, messages: list[dict], temperature: float = 0.2,
             max_tokens: int = 1500) -> str:
    r = client.chat.completions.create(model=modele, messages=messages,
                                       temperature=temperature, max_tokens=max_tokens)
    return r.choices[0].message.content or ""


def extraire_json(texte: str) -> dict | None:
    """Récupère le premier objet JSON d'une réponse de modèle (tolère le texte autour)."""
    texte = re.sub(r"```(?:json)?", "", texte or "")
    debut = texte.find("{")
    while debut != -1:
        profondeur = 0
        for i in range(debut, len(texte)):
            if texte[i] == "{":
                profondeur += 1
            elif texte[i] == "}":
                profondeur -= 1
                if profondeur == 0:
                    try:
                        return json.loads(texte[debut:i + 1])
                    except json.JSONDecodeError:
                        break
        debut = texte.find("{", debut + 1)
    return None


# ---------------------------------------------------------------------------
# 4. Sauvegarde / chargement des données
# ---------------------------------------------------------------------------
def sauvegarder(df_pages, df_passages, embeddings, meta: dict, dossier: Path = DATA_DIR):
    dossier = Path(dossier)
    dossier.mkdir(parents=True, exist_ok=True)
    for vieux in dossier.glob("embeddings_*.npy"):
        vieux.unlink()
    pages = df_pages.copy()
    pages["categories"] = pages["categories"].apply(lambda x: json.dumps(list(x), ensure_ascii=False))
    pages["fields"] = pages["fields"].apply(lambda x: json.dumps(x, ensure_ascii=False))
    pages.drop(columns=["text"], errors="ignore").to_parquet(dossier / "pages.parquet", index=False)
    df_passages.to_parquet(dossier / "passages.parquet", index=False)

    emb16 = embeddings.astype(np.float16)
    lignes_par_fichier = max(1, SHARD_MAX_BYTES // max(1, emb16.shape[1] * 2))
    n = 0
    for n, i in enumerate(range(0, len(emb16), lignes_par_fichier)):
        np.save(dossier / f"embeddings_{n:03d}.npy", emb16[i:i + lignes_par_fichier])
    meta = {**meta, "nb_pages": len(df_pages), "nb_passages": len(df_passages),
            "dimension": int(emb16.shape[1]) if emb16.ndim == 2 else 0,
            "nb_fichiers_embeddings": n + 1}
    (dossier / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def charger(dossier: Path = DATA_DIR):
    dossier = Path(dossier)
    meta = json.loads((dossier / "meta.json").read_text(encoding="utf-8"))
    pages = pd.read_parquet(dossier / "pages.parquet")
    pages["categories"] = pages["categories"].apply(json.loads)
    pages["fields"] = pages["fields"].apply(json.loads)
    passages = pd.read_parquet(dossier / "passages.parquet")
    fichiers = sorted(dossier.glob("embeddings_*.npy"))
    emb = np.concatenate([np.load(f) for f in fichiers]).astype(np.float32)
    return pages, passages, emb, meta


# ---------------------------------------------------------------------------
# 5. Recherche dans le texte (sémantique + bonus si le titre correspond)
# ---------------------------------------------------------------------------
def rechercher_passages(question_vec: np.ndarray, embeddings: np.ndarray,
                        passages: pd.DataFrame, question: str = "", k: int = 8,
                        max_par_page: int = 2) -> pd.DataFrame:
    scores = embeddings @ question_vec.astype(np.float32)
    mots = {m for m in re.findall(r"\w{4,}", normaliser(question))}
    if mots:
        titres = passages["title"].map(normaliser)
        bonus = titres.map(lambda t: sum(m in t for m in mots)).to_numpy()
        scores = scores + 0.03 * np.minimum(bonus, 3)
    ordre = np.argsort(-scores)
    retenus, par_page = [], {}
    for idx in ordre:
        t = passages.iloc[idx]["title"]
        if par_page.get(t, 0) >= max_par_page:
            continue
        par_page[t] = par_page.get(t, 0) + 1
        retenus.append(idx)
        if len(retenus) >= k:
            break
    res = passages.iloc[retenus].copy()
    res["score"] = scores[retenus]
    return res


# ---------------------------------------------------------------------------
# 6. Requêtes structurées sur les fiches (listes, comptages, répartitions)
# ---------------------------------------------------------------------------
def schema_fiches(pages: pd.DataFrame, min_pages: int = 5, seuil: float = 0.25,
                  max_categories: int = 40, max_champs: int = 15) -> dict:
    """Catégories principales et champs les plus remplis : le « plan » montré au modèle."""
    compte: dict[str, int] = {}
    for cats in pages["categories"]:
        for c in cats:
            compte[c] = compte.get(c, 0) + 1
    schema = {}
    for cat, n in sorted(compte.items(), key=lambda x: -x[1])[:max_categories]:
        if n < min_pages:
            continue
        sous = pages[pages["categories"].apply(lambda cs: cat in cs)]
        freq: dict[str, int] = {}
        for f in sous["fields"]:
            for cle in f:
                freq[cle] = freq.get(cle, 0) + 1
        champs = [c for c, v in sorted(freq.items(), key=lambda x: -x[1]) if v / n >= seuil]
        schema[cat] = {"nb_pages": n, "champs": champs[:max_champs]}
    return schema


def _valeur(fiche: dict, champ: str) -> str:
    if champ in fiche:
        return fiche[champ]
    cible = normaliser(champ)
    for cle, v in fiche.items():
        if normaliser(cle) == cible:
            return v
    return ""


def executer_requete(pages: pd.DataFrame, req: dict) -> dict:
    """
    req = {"category": "Acteur" | null,
           "filters": [{"field": "Theme" | "any", "contains": "eXtrême Défi"}],
           "operation": "list" | "count" | "group_count",
           "group_by": "Theme" | null}
    """
    df = pages
    cat = req.get("category")
    if cat:
        cible = normaliser(cat)
        df = df[df["categories"].apply(lambda cs: any(normaliser(c) == cible for c in cs))]
    for f in req.get("filters") or []:
        champ, mot = f.get("field") or "any", normaliser(f.get("contains", ""))
        if not mot:
            continue
        if champ == "any":
            masque = df.apply(lambda r: mot in normaliser(r["title"]) or
                              any(mot in normaliser(v) for v in r["fields"].values()) or
                              any(mot in normaliser(c) for c in r["categories"]), axis=1)
        elif normaliser(champ) in ("titre", "title"):
            masque = df["title"].map(lambda t: mot in normaliser(t))
        else:
            masque = df["fields"].apply(lambda fi: mot in normaliser(_valeur(fi, champ)))
        df = df[masque] if len(df) else df

    op = req.get("operation") or "list"
    resultat = {"operation": op, "nombre": int(len(df))}
    if op == "group_count" and req.get("group_by"):
        compte: dict[str, int] = {}
        for fi in df["fields"]:
            brut = _valeur(fi, req["group_by"])
            valeurs = [v.strip() for v in re.split(r"[,;]", brut) if v.strip()] or ["(non renseigné)"]
            for v in valeurs:
                compte[v] = compte.get(v, 0) + 1
        resultat["repartition"] = dict(sorted(compte.items(), key=lambda x: -x[1]))

    desc_cles = ("shortDescription", "ShortDescription", "description", "Description")
    lignes = []
    for _, r in df.iterrows():
        desc = next((r["fields"][c] for c in desc_cles if r["fields"].get(c)), "")
        ligne = {"Page": r["title"], "Lien": r["url"], "Description": desc[:300]}
        if req.get("group_by"):
            ligne[req["group_by"]] = _valeur(r["fields"], req["group_by"])
        lignes.append(ligne)
    resultat["table"] = pd.DataFrame(lignes)
    return resultat


# ---------------------------------------------------------------------------
# 7. Orchestration : router la question, chercher, répondre
# ---------------------------------------------------------------------------
PROMPT_ROUTEUR = """Tu aides à interroger le wiki de l'eXtrême Défi (XD) de l'ADEME et de la Fabrique des Mobilités.
Deux sources sont disponibles :
1. "structured" : une table des fiches du wiki (une ligne par page, avec ses catégories et les champs de sa fiche).
   À utiliser pour lister, compter, répartir, filtrer des acteurs, projets, véhicules, ressources…
2. "text" : une recherche dans le texte des pages. À utiliser pour les questions ouvertes (comment, pourquoi, retours d'expérience…).
Tu peux demander les deux ("both") si la question combine une liste et une explication.

Catégories et champs disponibles (nombre de pages entre parenthèses) :
{schema}

Réponds UNIQUEMENT avec un objet JSON de cette forme :
{{"mode": "structured" | "text" | "both",
  "category": "<nom exact d'une catégorie ci-dessus ou null>",
  "filters": [{{"field": "<nom d'un champ, ou 'any' pour chercher partout, ou 'title'>", "contains": "<texte>"}}],
  "operation": "list" | "count" | "group_count",
  "group_by": "<nom d'un champ ou null>",
  "search_query": "<reformulation de la question pour la recherche dans le texte>"}}"""

PROMPT_REPONSE = """Tu es un assistant qui répond aux questions sur le wiki de l'eXtrême Défi (XD) de l'ADEME
et de la Fabrique des Mobilités, en t'appuyant UNIQUEMENT sur les éléments fournis ci-dessous.
Règles :
- Réponds en français, de façon claire et concise.
- Pour les nombres et listes, utilise exclusivement le « Résultat de la table des fiches » (ne recompte pas toi-même).
- Cite tes sources sous la forme [1], [2]… correspondant aux passages numérotés.
- Si les éléments ne permettent pas de répondre, dis-le franchement et suggère une reformulation.
- Ne fabrique jamais de nom de projet, d'acteur ou de chiffre."""


def formater_schema(schema: dict) -> str:
    return "\n".join(f"- {cat} ({v['nb_pages']}) : {', '.join(v['champs']) or 'aucun champ'}"
                     for cat, v in schema.items())


def router(client, modele: str, question: str, schema: dict, historique: list[dict] | None = None) -> dict:
    contexte = ""
    if historique:
        derniers = historique[-4:]
        contexte = "Échanges précédents :\n" + "\n".join(
            f"{m['role']}: {m['content'][:400]}" for m in derniers) + "\n\n"
    brut = discuter(client, modele, [
        {"role": "system", "content": PROMPT_ROUTEUR.format(schema=formater_schema(schema))},
        {"role": "user", "content": contexte + "Question : " + question},
    ], temperature=0, max_tokens=400)
    plan = extraire_json(brut) or {}
    plan.setdefault("mode", "text")
    if plan["mode"] not in ("structured", "text", "both"):
        plan["mode"] = "text"
    plan.setdefault("search_query", question)
    plan["search_query"] = plan["search_query"] or question
    return plan


def repondre(client, modele_chat: str, modele_emb: str, question: str,
             pages: pd.DataFrame, passages: pd.DataFrame, embeddings: np.ndarray,
             schema: dict, historique: list[dict] | None = None, k: int = 8) -> dict:
    plan = router(client, modele_chat, question, schema, historique)
    contexte, structure, extraits = [], None, pd.DataFrame()

    if plan["mode"] in ("structured", "both"):
        structure = executer_requete(pages, plan)
        t = structure["table"]
        bloc = f"Résultat de la table des fiches : {structure['nombre']} page(s) trouvée(s).\n"
        if structure.get("repartition"):
            bloc += "Répartition : " + json.dumps(structure["repartition"], ensure_ascii=False)[:2500] + "\n"
        if len(t):
            bloc += t.drop(columns=["Lien"]).head(60).to_csv(index=False)[:6000]
            if len(t) > 60:
                bloc += f"\n(… {len(t) - 60} autres pages non affichées)"
        contexte.append(bloc)
        if structure["nombre"] == 0:  # filet de sécurité : on cherche aussi dans le texte
            plan["mode"] = "both"

    if plan["mode"] in ("text", "both"):
        vec = calculer_embeddings(client, modele_emb, [plan["search_query"]], pause=0)[0]
        extraits = rechercher_passages(vec, embeddings, passages, question=question, k=k)
        contexte.append("Passages du wiki :\n" + "\n\n".join(
            f"[{i}] ({r['title']}) {r['chunk']}" for i, (_, r) in enumerate(extraits.iterrows(), 1)))

    messages = [{"role": "system", "content": PROMPT_REPONSE}]
    for m in (historique or [])[-6:]:
        messages.append({"role": m["role"], "content": m["content"][:1500]})
    messages.append({"role": "user", "content": "\n\n".join(contexte) + f"\n\nQuestion : {question}"})
    reponse = discuter(client, modele_chat, messages)
    return {"reponse": reponse, "plan": plan, "structure": structure, "passages": extraits}
