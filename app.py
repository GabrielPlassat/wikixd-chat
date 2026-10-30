"""
Discuter avec le wiki de l'eXtrême Défi — app Streamlit (recherche hybride + API Albert).
Lancement local : streamlit run app.py
Clé API : dans les secrets Streamlit, ALBERT_API_KEY = "..."
"""
import streamlit as st

import core

st.set_page_config(page_title="Discuter avec le wiki XD", page_icon="🚲", layout="wide")


@st.cache_resource(show_spinner="Chargement des données du wiki…")
def donnees():
    pages, passages, emb, meta = core.charger()
    return pages, passages, emb, meta, core.schema_fiches(pages)


@st.cache_resource
def client(cle):
    return core.client_albert(cle)


@st.cache_data(ttl=3600, show_spinner=False)
def liste_modeles(cle):
    return core.modeles_albert(client(cle))


# --- Clé API -----------------------------------------------------------------
cle = st.secrets.get("ALBERT_API_KEY", "") if hasattr(st, "secrets") else ""
if not cle:
    cle = st.sidebar.text_input("Clé API Albert", type="password")
if not cle:
    st.info("Renseignez une clé API Albert (dans les secrets Streamlit ou dans la barre latérale).")
    st.stop()

try:
    pages, passages, emb, meta, schema = donnees()
except FileNotFoundError:
    st.error("Données absentes : lancez d'abord le notebook « construire_donnees.ipynb » "
             "puis déposez le dossier data/ dans le dépôt GitHub.")
    st.stop()

# --- Barre latérale ----------------------------------------------------------
with st.sidebar:
    st.header("Réglages")
    try:
        modeles = liste_modeles(cle)
    except Exception as e:  # noqa: BLE001
        st.error(f"Impossible de joindre Albert : {e}")
        st.stop()
    choix = modeles["chat"] or ["albert-large"]
    defaut = next((i for i, m in enumerate(choix) if "gpt-oss-120b" in m or "large" in m.lower()), 0)
    modele_chat = st.selectbox("Modèle de réponse", choix, index=defaut)
    modele_emb = meta["modele_embeddings"]
    if modele_emb not in modeles["embeddings"]:
        st.warning(f"Le modèle d'embeddings des données ({modele_emb}) n'est plus listé par Albert : "
                   "relancez le notebook.")
    k = st.slider("Passages consultés", 4, 15, 8)
    voir_plan = st.toggle("Montrer comment la question est traitée", value=False)
    st.divider()
    st.caption(f"Wiki aspiré le {meta['date_aspiration']} · {meta['nb_pages']} pages · "
               f"{meta['nb_passages']} passages")
    if meta.get("mode_test"):
        st.warning("Données en mode test (wiki partiel).")
    with st.expander("Catégories disponibles"):
        for cat, v in schema.items():
            st.markdown(f"**{cat}** ({v['nb_pages']})")
    if st.button("Nouvelle conversation"):
        st.session_state.messages = []
        st.rerun()

# --- Conversation ------------------------------------------------------------
st.title("🚲 Discuter avec le wiki de l'eXtrême Défi")
st.caption("Réponses générées par Albert à partir du contenu du wiki de la Fabrique des Mobilités. "
           "Vérifiez les sources citées.")

if "messages" not in st.session_state:
    st.session_state.messages = []


def afficher_details(det):
    if not det:
        return
    if voir_plan and det.get("plan"):
        with st.expander("Traitement de la question"):
            st.json(det["plan"])
    s = det.get("structure")
    if s is not None and len(s["table"]):
        with st.expander(f"Fiches trouvées ({s['nombre']})"):
            if s.get("repartition"):
                st.bar_chart(s["repartition"], horizontal=True)
            st.dataframe(s["table"], hide_index=True, width="stretch",
                         column_config={"Lien": st.column_config.LinkColumn("Lien", display_text="ouvrir")})
    p = det.get("passages")
    if p is not None and len(p):
        with st.expander("Sources"):
            for i, (_, r) in enumerate(p.iterrows(), 1):
                st.markdown(f"**[{i}]** [{r['title']}]({r['url']})")


for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        afficher_details(m.get("details"))

if not st.session_state.messages:
    st.markdown("**Exemples :** Combien de véhicules ont participé à l'XD ? · "
                "Quels acteurs travaillent sur la réparabilité ? · "
                "Comment financer un projet de véli ?")

if question := st.chat_input("Posez votre question sur le wiki…"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        historique = [{"role": m["role"], "content": m["content"]}
                      for m in st.session_state.messages[:-1]]
        with st.spinner("Recherche dans le wiki…"):
            try:
                res = core.repondre(client(cle), modele_chat, modele_emb, question,
                                    pages, passages, emb, schema, historique, k=k)
            except Exception as e:  # noqa: BLE001
                st.error(f"Erreur : {e}")
                st.stop()
        st.markdown(res["reponse"])
        details = {"plan": res["plan"], "structure": res["structure"], "passages": res["passages"]}
        afficher_details(details)
    st.session_state.messages.append({"role": "assistant", "content": res["reponse"],
                                      "details": details})
