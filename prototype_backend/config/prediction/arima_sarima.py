# ============================================================
# prediction/arima_sarima.py
# Prévision de la demande par ARIMA et SARIMA
# Phase 2 : Intelligence — benchmark mathématique (cf. rapport §2.6)
# ============================================================
 
import numpy as np
import pandas as pd
import warnings
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error
import sys, os
 
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app_config.settings import (
    TRAIN_RATIO, HORIZON_PREDICTION,
    ARIMA_ORDER, SARIMA_ORDER, SARIMA_SEASONAL
)
 
warnings.filterwarnings("ignore")
 
 
# ──────────────────────────────────────────────────────────────
# Utilitaire : test de stationnarité ADF
# ──────────────────────────────────────────────────────────────
def tester_stationnarite(serie: pd.Series) -> dict:
    """
    Test de Dickey-Fuller augmenté.
    H0 : la série a une racine unitaire (non stationnaire).
    """
    result = adfuller(serie.dropna(), autolag="AIC")
    return {
        "ADF_stat":   round(float(result[0]), 4),
        "p_value":    round(float(result[1]), 4),
        "stationnaire": result[1] < 0.05,
        "nb_lags":    result[2],
        "n_obs":      result[3]
    }
 
 
def detecter_saisonnalite(serie: pd.Series, periode: int = 12) -> bool:
    """
    Détecte grossièrement la saisonnalité par autocorrélation à lag=période.
    Retourne True si saisonnalité significative.
    """
    if len(serie) < 2 * periode:
        return False
    acf_lag = serie.autocorr(lag=periode)
    return abs(acf_lag) > 0.3
 
 
# ──────────────────────────────────────────────────────────────
# Modèle ARIMA
# ──────────────────────────────────────────────────────────────
class ModeleARIMA:
    """
    Modèle ARIMA(p,d,q) pour séries temporelles univariées.
    Référence mathématique du rapport (§2.6).
    """
 
    def __init__(self, order: tuple = ARIMA_ORDER):
        self.order     = order
        self.resultat  = None
        self.est_entraine = False
        self.metriques = {}
 
    def entrainer(self, serie: pd.Series) -> dict:
        """Entraîne ARIMA sur 2/3 de la série, évalue sur 1/3."""
        n_train = int(len(serie) * TRAIN_RATIO)
        train   = serie.iloc[:n_train]
        test    = serie.iloc[n_train:]
 
        modele      = ARIMA(train, order=self.order)
        self.resultat = modele.fit()
        self.est_entraine = True
 
        # Prédiction sur période de test
        forecast = self.resultat.forecast(steps=len(test))
        forecast = np.clip(forecast, 0, None)
 
        mae  = float(mean_absolute_error(test.values, forecast))
        rmse = float(np.sqrt(mean_squared_error(test.values, forecast)))
 
        self.metriques = {
            "MAE":  round(mae, 3),
            "RMSE": round(rmse, 3),
            "AIC":  round(float(self.resultat.aic), 2),
            "BIC":  round(float(self.resultat.bic), 2),
            "ordre": self.order,
            "n_train": n_train,
            "n_test":  len(test)
        }
        return self.metriques
 
    def predire(self, horizon: int = HORIZON_PREDICTION,
                serie_complete: pd.Series = None) -> pd.Series:
        """
        Génère les prévisions futures.
        Si serie_complete est fournie, ré-entraîne sur toutes les données
        avant de prédire (meilleur pour les vraies prévisions futures).
        """
        if not self.est_entraine:
            raise RuntimeError("Entraîner le modèle avant de prédire.")
 
        if serie_complete is not None:
            modele_final = ARIMA(serie_complete, order=self.order).fit()
        else:
            modele_final = self.resultat
 
        forecast = modele_final.forecast(steps=horizon)
        forecast = np.clip(forecast, 0, None).round().astype(int)
 
        # Index temporel des prévisions
        if hasattr(serie_complete, "index") and hasattr(serie_complete.index, "freq"):
            last = serie_complete.index[-1]
            idx  = pd.date_range(start=last, periods=horizon + 1, freq="MS")[1:]
        else:
            idx = np.arange(len(serie_complete), len(serie_complete) + horizon)
 
        return pd.Series(forecast.values, index=idx, name="prediction_arima")
 
    def get_fitted_values(self, serie: pd.Series) -> pd.Series:
        """Valeurs ajustées sur les données d'entraînement (pour graphe)."""
        n_train = int(len(serie) * TRAIN_RATIO)
        return pd.Series(
            np.clip(self.resultat.fittedvalues, 0, None).round().astype(int),
            index=serie.index[:n_train],
            name="fitted_arima"
        )
 
 
# ──────────────────────────────────────────────────────────────
# Modèle SARIMA
# ──────────────────────────────────────────────────────────────
class ModeleSARIMA:
    """
    Modèle SARIMA(p,d,q)(P,D,Q,s) — supérieur à ARIMA pour les
    séries saisonnières (démontré dans le rapport §2.7.6).
    """
 
    def __init__(self, order: tuple = SARIMA_ORDER,
                 seasonal_order: tuple = SARIMA_SEASONAL):
        self.order          = order
        self.seasonal_order = seasonal_order
        self.resultat       = None
        self.est_entraine   = False
        self.metriques      = {}
 
    def entrainer(self, serie: pd.Series) -> dict:
        """Entraîne SARIMA sur 2/3 de la série, évalue sur 1/3."""
        n_train = int(len(serie) * TRAIN_RATIO)
        train   = serie.iloc[:n_train]
        test    = serie.iloc[n_train:]
 
        modele = SARIMAX(
            train,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        self.resultat    = modele.fit(disp=False)
        self.est_entraine = True
 
        # Prédiction sur période de test
        forecast = self.resultat.forecast(steps=len(test))
        forecast = np.clip(forecast, 0, None)
 
        mae  = float(mean_absolute_error(test.values, forecast))
        rmse = float(np.sqrt(mean_squared_error(test.values, forecast)))
 
        self.metriques = {
            "MAE":            round(mae, 3),
            "RMSE":           round(rmse, 3),
            "AIC":            round(float(self.resultat.aic), 2),
            "BIC":            round(float(self.resultat.bic), 2),
            "ordre":          self.order,
            "ordre_saisonnier": self.seasonal_order,
            "n_train":        n_train,
            "n_test":         len(test)
        }
        return self.metriques
 
    def predire(self, horizon: int = HORIZON_PREDICTION,
                serie_complete: pd.Series = None) -> pd.Series:
        """Génère les prévisions futures."""
        if not self.est_entraine:
            raise RuntimeError("Entraîner le modèle avant de prédire.")
 
        if serie_complete is not None:
            modele_final = SARIMAX(
                serie_complete,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            ).fit(disp=False)
        else:
            modele_final = self.resultat
 
        forecast = modele_final.forecast(steps=horizon)
        forecast = np.clip(forecast, 0, None).round().astype(int)
 
        if serie_complete is not None and hasattr(serie_complete.index, "freq"):
            last = serie_complete.index[-1]
            idx  = pd.date_range(start=last, periods=horizon + 1, freq="MS")[1:]
        else:
            base = len(serie_complete) if serie_complete is not None else 0
            idx  = np.arange(base, base + horizon)
 
        return pd.Series(forecast.values, index=idx, name="prediction_sarima")
 
    def get_fitted_values(self, serie: pd.Series) -> pd.Series:
        """Valeurs ajustées SARIMA sur données d'entraînement."""
        n_train = int(len(serie) * TRAIN_RATIO)
        fitted  = self.resultat.fittedvalues
        return pd.Series(
            np.clip(fitted, 0, None).round().astype(int),
            index=serie.index[:len(fitted)],
            name="fitted_sarima"
        )
 
 
# ──────────────────────────────────────────────────────────────
# Comparaison automatique des modèles (cf. Figure 2.6 rapport)
# ──────────────────────────────────────────────────────────────
def comparer_modeles(serie: pd.Series) -> dict:
    """
    Entraîne ARIMA et SARIMA, retourne un comparatif complet
    avec le meilleur modèle recommandé (comme dans le rapport §2.7.6).
    """
    print("  → Test de stationnarité...")
    adf = tester_stationnarite(serie)
 
    print("  → Détection saisonnalité...")
    saisonnier = detecter_saisonnalite(serie)
 
    print("  → Entraînement ARIMA...")
    arima = ModeleARIMA()
    m_arima = arima.entrainer(serie)
 
    print("  → Entraînement SARIMA...")
    sarima = ModeleSARIMA()
    m_sarima = sarima.entrainer(serie)
 
    # Recommandation basée sur RMSE (comme dans le rapport)
    meilleur = "SARIMA" if m_sarima["RMSE"] <= m_arima["RMSE"] else "ARIMA"
 
    return {
        "stationnarite": adf,
        "saisonnalite_detectee": saisonnier,
        "ARIMA":  m_arima,
        "SARIMA": m_sarima,
        "meilleur_modele": meilleur,
        "objets": {"arima": arima, "sarima": sarima}
    }