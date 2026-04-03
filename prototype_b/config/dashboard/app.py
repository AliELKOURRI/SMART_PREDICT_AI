
Copier

# ============================================================
# dashboard/app.py
# Interface Streamlit — Tableau de bord intelligent
# Phase 3 : Décision (cf. rapport §2.7.3 & slide 14)
# Lancement : streamlit run dashboard/app.py
# ============================================================
 
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import sys, os, time
 
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
 
from app_config.settings import (
    PAGE_TITLE, PAGE_ICON, SEUIL_ALERTE, SEUIL_CRITIQUE,
    HORIZON_PREDICTION, REFRESH_SECONDES, POSITION_ENTREE, POSITION_SORTIE
)
from helpers.data_manager   import (
    charger_stock, charger_historique, charger_commandes,
    enregistrer_entree, enregistrer_sortie,
    detecter_alertes, statistiques_stock, get_liste_produits
)
from prediction.prediction_engine import MoteurPrediction
from robot.mission_manager    import GestionnaireMissions
 
# ──────────────────────────────────────────────────────────────
# Configuration page
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)
 
# ──────────────────────────────────────────────────────────────
# Fonctions utilitaires
# ──────────────────────────────────────────────────────────────
 
@st.cache_data(ttl=60)
def get_stock():
    return charger_stock()
 
@st.cache_data(ttl=300)
def get_historique(produit):
    return charger_historique(produit=produit)
 
@st.cache_data(ttl=60)
def get_stats():
    return statistiques_stock()
 
def couleur_stock(qte):
    if qte <= SEUIL_CRITIQUE: return "🔴"
    if qte <= SEUIL_ALERTE:   return "🟡"
    return "🟢"
 
 
# ──────────────────────────────────────────────────────────────
# SIDEBAR — Navigation
# ──────────────────────────────────────────────────────────────
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/39/"
                 "Centrale_Casablanca_logo.png/320px-Centrale_Casablanca_logo.png",
                 width=160)
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Aller à :",
    ["🏠 Tableau de bord",
     "📊 Prévisions & Modèles",
     "📦 Gestion des stocks",
     "🤖 Robot & Missions",
     "📋 Historique commandes"]
)
 
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Projet PLBD 20 — 2025-2026**")
st.sidebar.markdown("*Vers un entrepôt intelligent*")
 
 
# ══════════════════════════════════════════════════════════════
# PAGE 1 : TABLEAU DE BORD PRINCIPAL
# ══════════════════════════════════════════════════════════════
if page == "🏠 Tableau de bord":
    st.title("🏭 Entrepôt Intelligent — Tableau de bord")
 
    # Alertes critiques en haut
    alertes = detecter_alertes()
    if not alertes.empty:
        critiques = alertes[alertes["niveau"] == "critique"]
        if not critiques.empty:
            for _, row in critiques.iterrows():
                st.error(f"🚨 ALERTE CRITIQUE — {row['produit']} : "
                         f"seulement {row['quantite']} unités en stock !")
        warns = alertes[alertes["niveau"] == "alerte"]
        for _, row in warns.iterrows():
            st.warning(f"⚠️ Alerte — {row['produit']} : {row['quantite']} unités (seuil : {SEUIL_ALERTE})")
 
    # ── KPIs ──────────────────────────────────────────────────
    st.subheader("Indicateurs clés")
    stats = get_stats()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📦 Total unités en stock", stats["total_unites"])
    col2.metric("✅ Produits OK",           stats["produits_ok"])
    col3.metric("⚠️ Produits en alerte",    stats["produits_alerte"],
                delta=f"-{stats['produits_alerte']}" if stats["produits_alerte"] > 0 else None,
                delta_color="inverse")
    col4.metric("🛒 Commandes (30 jours)",  stats["commandes_30j"])
 
    st.markdown("---")
 
    # ── État des stocks ────────────────────────────────────────
    col_left, col_right = st.columns([2, 1])
 
    with col_left:
        st.subheader("État des stocks par produit")
        stock_df = get_stock()
        fig = go.Figure()
        for _, row in stock_df.iterrows():
            pct   = min(100, int(row["quantite"] / 100 * 100))
            color = "#e74c3c" if row["quantite"] <= SEUIL_CRITIQUE else \
                    "#f39c12" if row["quantite"] <= SEUIL_ALERTE else "#2ecc71"
            fig.add_trace(go.Bar(
                name=row["produit"],
                x=[row["produit"]],
                y=[row["quantite"]],
                marker_color=color,
                text=[f"{row['quantite']} unités"],
                textposition="outside"
            ))
 
        fig.add_hline(y=SEUIL_ALERTE,   line_dash="dash",
                      line_color="#f39c12", annotation_text="Seuil alerte")
        fig.add_hline(y=SEUIL_CRITIQUE, line_dash="dot",
                      line_color="#e74c3c",  annotation_text="Seuil critique")
        fig.update_layout(
            showlegend=False, height=350,
            yaxis_title="Quantité (unités)",
            margin=dict(t=20, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)
 
    with col_right:
        st.subheader("Résumé stock")
        for _, row in stock_df.iterrows():
            ico  = couleur_stock(row["quantite"])
            pct  = min(100, max(0, int(row["quantite"])))
            st.markdown(f"**{ico} {row['produit']}** — {row['quantite']} u.")
            st.progress(min(1.0, row["quantite"] / 100))
 
    # ── Évolution commandes 30 jours ───────────────────────────
    st.subheader("Commandes des 30 derniers jours")
    cmds = charger_commandes()
    cmds_recent = cmds[cmds["date"] >= pd.Timestamp.now() - pd.Timedelta(days=30)]
    if not cmds_recent.empty:
        cmds_grouped = cmds_recent.groupby(["date", "produit"])["quantite"].sum().reset_index()
        fig2 = px.bar(cmds_grouped, x="date", y="quantite", color="produit",
                      title="Sorties quotidiennes par produit",
                      labels={"quantite": "Unités commandées", "date": "Date"})
        fig2.update_layout(height=300, margin=dict(t=40, b=20))
        st.plotly_chart(fig2, use_container_width=True)
 
 
# ══════════════════════════════════════════════════════════════
# PAGE 2 : PRÉVISIONS & MODÈLES
# ══════════════════════════════════════════════════════════════
elif page == "📊 Prévisions & Modèles":
    st.title("📊 Prévisions de la demande")
    st.markdown(
        "Comparaison des modèles **Régression Linéaire**, **ARIMA** et **SARIMA** "
        "(cf. §2.6 et §2.7.6 du rapport)."
    )
 
    produits = get_liste_produits()
    produit_sel = st.selectbox("Sélectionner un produit", produits)
    horizon     = st.slider("Horizon de prévision (mois)", 3, 24, HORIZON_PREDICTION)
 
    if st.button("🚀 Lancer l'analyse prédictive", type="primary"):
        with st.spinner(f"Entraînement des modèles pour {produit_sel}..."):
            try:
                moteur = MoteurPrediction(produit_sel)
                moteur.charger_serie()
                metriques = moteur.entrainer_tous()
                df_pred   = moteur.predire(horizon=horizon)
                reappro   = moteur.recommander_reappro()
 
                st.session_state["metriques"]  = metriques
                st.session_state["df_pred"]    = df_pred
                st.session_state["reappro"]    = reappro
                st.session_state["moteur"]     = moteur
                st.session_state["produit_sel"] = produit_sel
                st.success("✅ Analyse terminée !")
            except Exception as e:
                st.error(f"❌ Erreur : {e}")
 
    # Affichage des résultats
    if "df_pred" in st.session_state and st.session_state.get("produit_sel") == produit_sel:
        metriques = st.session_state["metriques"]
        df_pred   = st.session_state["df_pred"]
        reappro   = st.session_state["reappro"]
        moteur    = st.session_state["moteur"]
 
        st.markdown("---")
 
        # ── Métriques comparatives ──────────────────────────────
        st.subheader("Comparatif des performances (sur données de test)")
        col1, col2, col3 = st.columns(3)
 
        with col1:
            st.markdown("**📈 Régression Linéaire**")
            st.metric("RMSE", metriques["regression"]["RMSE"])
            st.metric("MAE",  metriques["regression"]["MAE"])
            st.metric("R²",   metriques["regression"]["R²"])
 
        with col2:
            st.markdown("**📉 ARIMA**")
            st.metric("RMSE", metriques["arima"]["RMSE"])
            st.metric("MAE",  metriques["arima"]["MAE"])
            st.metric("AIC",  metriques["arima"]["AIC"])
 
        with col3:
            st.markdown("**⭐ SARIMA** ← recommandé si saisonnier")
            st.metric("RMSE", metriques["sarima"]["RMSE"])
            st.metric("MAE",  metriques["sarima"]["MAE"])
            st.metric("AIC",  metriques["sarima"]["AIC"])
 
        meilleur = metriques["meilleur"]
        st.info(f"🏆 **Meilleur modèle recommandé : {meilleur}** "
                f"(RMSE le plus faible sur données de test)")
 
        st.markdown("---")
 
        # ── Graphique prévisions (réplique Figure 2.6) ──────────
        st.subheader(f"Prévisions — {produit_sel} (prochains {horizon} mois)")
        serie = moteur.serie
 
        fig = go.Figure()
        # Historique
        fig.add_trace(go.Scatter(
            x=serie.index, y=serie.values,
            mode="lines", name="Historique réel",
            line=dict(color="#2c3e50", width=2)
        ))
        # Prévisions
        couleurs = {"regression": "#3498db", "arima": "#e74c3c", "sarima": "#9b59b6"}
        noms     = {"regression": "Régression",
                    "arima":      "ARIMA",
                    "sarima":     "SARIMA"}
        for col, couleur in couleurs.items():
            fig.add_trace(go.Scatter(
                x=df_pred["date"], y=df_pred[col],
                mode="lines+markers", name=noms[col],
                line=dict(color=couleur, width=2, dash="dash"),
                marker=dict(size=5)
            ))
 
        # Zone de prévision
        fig.add_vrect(
            x0=df_pred["date"].iloc[0], x1=df_pred["date"].iloc[-1],
            fillcolor="lightgrey", opacity=0.2,
            layer="below", line_width=0,
            annotation_text="Zone de prévision", annotation_position="top left"
        )
        fig.update_layout(
            height=450,
            xaxis_title="Date",
            yaxis_title="Demande (unités)",
            legend=dict(orientation="h", y=-0.2),
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)
 
        # ── Recommandation réapprovisionnement ──────────────────
        st.markdown("---")
        st.subheader("🔄 Recommandation de réapprovisionnement")
 
        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown(f"""
| Paramètre | Valeur |
|-----------|--------|
| Stock actuel | **{reappro['stock_actuel']} unités** |
| Demande prévue (30 j) | **{reappro['demande_prevue_J30']} unités** |
| Déficit estimé | **{reappro['deficit']} unités** |
| Quantité à commander | **{reappro['reappro_conseille']} unités** |
| Fournisseur | {reappro['fournisseur']} |
| Délai livraison | {reappro['delai_livraison']} jours |
""")
        with col_r:
            if reappro["alerte_critique"]:
                st.error(f"🚨 Stock CRITIQUE ! Commander {reappro['reappro_conseille']} unités immédiatement.")
            elif reappro["deficit"] > 0:
                st.warning(f"⚠️ Déficit prévu. Commander {reappro['reappro_conseille']} unités.")
            else:
                st.success("✅ Stock suffisant pour couvrir la demande prévue.")
 
 
# ══════════════════════════════════════════════════════════════
# PAGE 3 : GESTION DES STOCKS
# ══════════════════════════════════════════════════════════════
elif page == "📦 Gestion des stocks":
    st.title("📦 Gestion opérationnelle des stocks")
 
    col1, col2 = st.columns(2)
 
    with col1:
        st.subheader("➕ Entrée de stock (livraison fournisseur)")
        produits   = get_liste_produits()
        prod_entree = st.selectbox("Produit", produits, key="entree_prod")
        qte_entree  = st.number_input("Quantité reçue", min_value=1, max_value=10000,
                                      value=50, key="entree_qte")
        if st.button("Valider l'entrée", type="primary", key="btn_entree"):
            try:
                res = enregistrer_entree(prod_entree, int(qte_entree))
                st.success(f"✅ Stock mis à jour : {prod_entree} → {res['quantite']} unités")
                st.cache_data.clear()
            except Exception as e:
                st.error(f"❌ {e}")
 
    with col2:
        st.subheader("➖ Sortie de stock (commande client)")
        prod_sortie = st.selectbox("Produit", produits, key="sortie_prod")
        qte_sortie  = st.number_input("Quantité sortie", min_value=1, max_value=1000,
                                       value=10, key="sortie_qte")
        if st.button("Valider la sortie", type="primary", key="btn_sortie"):
            try:
                res = enregistrer_sortie(prod_sortie, int(qte_sortie))
                st.success(f"✅ Sortie enregistrée : {prod_sortie} → {res['quantite']} unités")
                st.cache_data.clear()
            except Exception as e:
                st.error(f"❌ {e}")
 
    st.markdown("---")
    st.subheader("📋 Stock actuel")
    stock_df = charger_stock()
    # Ajout colonnes de visualisation
    stock_df["Niveau"] = stock_df["quantite"].apply(
        lambda q: "🔴 Critique" if q <= SEUIL_CRITIQUE else
                  "🟡 Alerte"   if q <= SEUIL_ALERTE   else "🟢 OK"
    )
    st.dataframe(
        stock_df[["produit", "quantite", "zone", "ligne", "colonne", "Niveau", "date_maj"]],
        use_container_width=True,
        hide_index=True
    )
 
 
# ══════════════════════════════════════════════════════════════
# PAGE 4 : ROBOT & MISSIONS
# ══════════════════════════════════════════════════════════════
elif page == "🤖 Robot & Missions":
    st.title("🤖 Contrôle du Robot — Algorithme A*")
    st.markdown(
        "Le robot navigue de manière autonome via l'algorithme A*, "
        "en évitant les zones congestionnées (heatmap) — cf. slide 15."
    )
 
    # Initialisation du gestionnaire (sans GPIO si pas Raspberry Pi)
    if "gestionnaire" not in st.session_state:
        with st.spinner("Initialisation du gestionnaire de missions..."):
            st.session_state["gestionnaire"] = GestionnaireMissions()
 
    gestionnaire = st.session_state["gestionnaire"]
 
    col1, col2 = st.columns([1, 2])
 
    with col1:
        st.subheader("Nouvelle mission")
        produits_dispo = get_liste_produits()
        produits_sel   = st.multiselect(
            "Produits à collecter", produits_dispo,
            default=[produits_dispo[2]] if len(produits_dispo) > 2 else produits_dispo[:1]
        )
 
        if st.button("🚀 Planifier & Simuler le chemin", type="primary"):
            if not produits_sel:
                st.warning("Sélectionner au moins un produit.")
            else:
                with st.spinner("Calcul du chemin A*..."):
                    try:
                        plan = gestionnaire.simuler_chemin(produits_sel)
                        st.session_state["plan_actuel"] = plan
                        st.session_state["produits_mission"] = produits_sel
                        st.success(
                            f"✅ Chemin calculé : {plan['distance_totale']} cases, "
                            f"{plan['nb_etapes']} étapes"
                        )
                    except Exception as e:
                        st.error(f"❌ {e}")
 
        if st.button("🤖 Exécuter sur le robot", type="secondary"):
            if "plan_actuel" not in st.session_state:
                st.warning("Planifier d'abord le chemin.")
            else:
                produits_m = st.session_state["produits_mission"]
                barre = st.progress(0)
                statut_txt = st.empty()
 
                def callback(etape, position, total):
                    barre.progress(etape / total)
                    statut_txt.markdown(
                        f"🤖 Étape **{etape}/{total}** — Position : `{position}`"
                    )
 
                with st.spinner("Mission en cours..."):
                    resultat = gestionnaire.lancer_mission(
                        produits_m, source="dashboard", callback_etape=callback
                    )
                barre.progress(1.0)
                if resultat.get("statut") == "succes":
                    st.success("✅ Mission accomplie !")
                else:
                    st.error(f"❌ Mission : {resultat.get('statut')}")
 
    with col2:
        st.subheader("Visualisation de la grille")
 
        if "plan_actuel" in st.session_state:
            plan   = st.session_state["plan_actuel"]
            grille = gestionnaire.grille
            chemin = set(plan["chemin_total"])
 
            # Construire figure Plotly de la grille
            fig = go.Figure()
            for l in range(grille.lignes):
                for c in range(grille.colonnes):
                    val = grille.grille[l][c]
                    pos = (l, c)
                    if pos == POSITION_ENTREE:
                        couleur, texte = "#2ecc71", "S"
                    elif pos == POSITION_SORTIE:
                        couleur, texte = "#e74c3c", "E"
                    elif pos in chemin:
                        couleur, texte = "#3498db", "→"
                    elif val == 1:
                        couleur, texte = "#7f8c8d", "█"
                    elif val == 2:
                        couleur, texte = "#f39c12", "P"
                    else:
                        couleur, texte = "#ecf0f1", ""
 
                    fig.add_trace(go.Scatter(
                        x=[c], y=[grille.lignes - 1 - l],
                        mode="markers+text",
                        marker=dict(color=couleur, size=22, symbol="square"),
                        text=texte, textfont=dict(size=9, color="white"),
                        showlegend=False, hoverinfo="skip"
                    ))
 
            fig.update_layout(
                height=400,
                xaxis=dict(showgrid=False, zeroline=False, range=[-0.5, grille.colonnes - 0.5]),
                yaxis=dict(showgrid=False, zeroline=False, range=[-0.5, grille.lignes - 0.5]),
                margin=dict(l=10, r=10, t=10, b=10),
                plot_bgcolor="#f8f9fa"
            )
            st.plotly_chart(fig, use_container_width=True)
 
            # Segments de la mission
            st.subheader("Segments de la mission")
            for seg in plan["segments"]:
                st.markdown(f"- **{seg['produit']}** : {seg['distance']} cases")
        else:
            st.info("Planifier une mission pour voir le chemin.")
 
    # Historique des missions
    st.markdown("---")
    st.subheader("📋 Historique des missions (Feedback)")
    hist = gestionnaire.get_historique_missions()
    if hist:
        df_hist = pd.DataFrame(hist)
        st.dataframe(
            df_hist[["id", "datetime", "produits", "statut", "distance", "duree_sec"]],
            use_container_width=True, hide_index=True
        )
    else:
        st.info("Aucune mission enregistrée.")
 
 
# ══════════════════════════════════════════════════════════════
# PAGE 5 : HISTORIQUE COMMANDES
# ══════════════════════════════════════════════════════════════
elif page == "📋 Historique commandes":
    st.title("📋 Historique des commandes")
 
    cmds = charger_commandes()
    cmds = cmds.sort_values("date", ascending=False).reset_index(drop=True)
 
    col1, col2 = st.columns(2)
    with col1:
        produits_filtre = st.multiselect(
            "Filtrer par produit", get_liste_produits(),
            default=get_liste_produits()
        )
    with col2:
        n_jours = st.slider("Derniers N jours", 7, 365, 60)
 
    date_limite = pd.Timestamp.now() - pd.Timedelta(days=n_jours)
    cmds_filtre = cmds[
        (cmds["produit"].isin(produits_filtre)) &
        (cmds["date"] >= date_limite)
    ]
 
    st.dataframe(cmds_filtre, use_container_width=True, hide_index=True)
    st.markdown(f"**{len(cmds_filtre)} commandes affichées**")
 
    # Graphe évolution
    if not cmds_filtre.empty:
        fig = px.line(
            cmds_filtre.groupby(["date", "produit"])["quantite"].sum().reset_index(),
            x="date", y="quantite", color="produit",
            title="Évolution des sorties dans la période sélectionnée"
        )
        st.plotly_chart(fig, use_container_width=True)