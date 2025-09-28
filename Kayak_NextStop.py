
import os
import math
from pathlib import Path
from typing import Tuple, Dict, Any
import pandas as pd
import requests
import streamlit as st
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import datetime as dt
from PIL import Image
import base64
import pathlib


# ==============================
# App config
# ==============================
st.set_page_config(
    page_title="Kayak NextStop",
    page_icon="assets/Favicon_Kayak_NextStop.png",  # favicon de l’onglet
    layout="wide"            # <<< important pour la largeur
)
# --- Header: grand logo centré + slogan stylé ---

def render_header(
    logo_path="assets/Logo_Kayak_NextStop.png",
    slogan="Quand météo et envie décident de votre prochaine destination.",
    logo_width=700,   # ← augmente la taille du logo (essaie 650–740 si besoin)
    slogan_px=32      # ← 30–35 px
):
    p = pathlib.Path(logo_path)
    left_pad, mid, right_pad = st.columns([1, 8, 1])  # centre le contenu
    with mid:
        if p.exists():
            st.image(logo_path, use_container_width=False, width=logo_width)
        else:
            st.markdown("<h1 style='text-align:center;'>Kayak NextStop</h1>", unsafe_allow_html=True)
            st.caption(f"⚠️ Logo introuvable à: {os.path.abspath(logo_path)}")

        # Slogan (centré, gris foncé, italique)
        st.markdown(
            f"<div style='text-align:center; margin-top:10px; "
            f"font-size:{slogan_px}px; font-weight:600; line-height:1.25; "
            f"color:#374151; font-style:italic;'>"
            f"{slogan}"
            f"</div>",
            unsafe_allow_html=True
        )

# Appel
render_header(
    logo_path="assets/Logo_Kayak_NextStop.png",
    slogan="Quand météo et envie décident de votre prochaine destination.",
    logo_width=700,
    slogan_px=25
)


MAP_HEIGHT = 440  # hauteur carte (px) adaptée mobile

# --- Defaults (session) ---
if "wind_unit" not in st.session_state:
    st.session_state["wind_unit"] = "km/h"  # défaut grand public

if "date_in" not in st.session_state:
    st.session_state["date_in"] = dt.date.today() + dt.timedelta(days=7)
if "date_out" not in st.session_state:
    st.session_state["date_out"] = st.session_state["date_in"] + dt.timedelta(days=3)


# ==============================
# Helpers / Config
# ==============================
def get_owm_key() -> str:
    # Priorité aux secrets Streamlit
    key = st.secrets.get("OPENWEATHER_API_KEY", None) if hasattr(st, "secrets") else None
    if not key:
        key = os.getenv("OPENWEATHER_API_KEY", "")
    return key

@st.cache_data(ttl=3600)
def load_curated_csv() -> pd.DataFrame:
    """
    Charge le CSV embarqué (chemin relatif) et retourne le dataframe brut.
    Le fichier attendu est: data/curated/destinations_scored.csv
    avec les colonnes: city,lat,lon,avg_t_day,rain_sum_mm,weather_score,hotel_score,score_final
    """
    data_csv = Path(__file__).parent / "data" / "curated" / "destinations_scored.csv"
    if not data_csv.exists():
        raise FileNotFoundError(f"CSV introuvable: {data_csv}")
    df = pd.read_csv(data_csv)
    expected = ["city","lat","lon","avg_t_day","rain_sum_mm","weather_score","hotel_score","score_final"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans le CSV: {missing}")
    return df

@st.cache_data(ttl=3600)
def make_views(df: pd.DataFrame):
    """Construit dprof (agrégé par ville) et dfw (villes uniques)."""
    dprof = (
        df.groupby(["city","lat","lon"], as_index=False)
          .agg(avg_t_day=("avg_t_day","mean"),
               rain_sum_mm=("rain_sum_mm","mean"),
               weather_score=("weather_score","mean"),
               hotel_score=("hotel_score","mean"),
               score_final=("score_final","mean"))
          .sort_values("city")
    )
    dfw = dprof[["city","lat","lon"]].copy()
    return dprof, dfw

# ------------------------------
# OSM helpers (hôtels proches)
# ------------------------------
@st.cache_data(ttl=3600)
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dlmb/2)**2
    return 2 * R * math.asin(math.sqrt(a))

@st.cache_data(ttl=1800)
def osm_hotels_within(lat, lon, radius_m=3000):
    """
    Retourne une liste de (lat, lon) d’hébergements OSM (tourism=hotel ou amenity=lodging)
    dans un rayon donné (mètres) autour (lat, lon). Utilise l’API Overpass (open data).
    """
    url = "https://overpass-api.de/api/interpreter"
    q = f"""
    [out:json][timeout:25];
    (
      node["tourism"="hotel"](around:{radius_m},{lat},{lon});
      node["amenity"="lodging"](around:{radius_m},{lat},{lon});
    );
    out center;
    """
    try:
        r = requests.post(url, data={"data": q}, timeout=25)
        r.raise_for_status()
        data = r.json().get("elements", [])
        pts = []
        for el in data:
            la = el.get("lat"); lo = el.get("lon")
            if la is not None and lo is not None:
                pts.append((float(la), float(lo)))
        return pts
    except Exception:
        return []

def make_hotel_links(city: str, date_in: dt.date, date_out: dt.date) -> dict:
    """Construit des liens directs vers des moteurs d'hôtels avec ville + dates."""
    city_q = requests.utils.quote(str(city))
    di, do = date_in.isoformat(), date_out.isoformat()
    return {
        "Kayak":         f"https://www.kayak.fr/hotels/{city_q}/{di}/{do}",
        "Google Hotels": f"https://www.google.com/travel/hotels/{city_q}?checkin={di}&checkout={do}",
        "Booking":       f"https://www.booking.com/searchresults.html?ss={city_q}&checkin={di}&checkout={do}",
    }

@st.cache_data(ttl=600)
def rain_prob_pct(lat: float, lon: float, api_key: str) -> int | None:
    """
    Probabilité moyenne de pluie (%) sur l'horizon OWM /forecast.
    Utilise le champ 'pop' (0..1) des pas 3h. Retourne un entier 0..100.
    """
    try:
        _, fc = fetch_weather(lat, lon, api_key, units="metric")
        pops = [float(x.get("pop", 0.0)) for x in fc.get("list", []) if isinstance(x, dict)]
        if not pops:
            return None
        return int(round(100 * (sum(pops) / len(pops))))
    except Exception:
        return None


# ------------------------------
# OpenWeatherMap helpers
# ------------------------------
@st.cache_data(ttl=900)
def fetch_weather(lat: float, lon: float, api_key: str, units: str = "metric") -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Récupère la météo actuelle et les prévisions 5 jours (3h) via OWM.
    Retourne (current_json, forecast_json)
    """
    params_base = {"lat": lat, "lon": lon, "appid": api_key, "units": units, "lang": "fr"}
    cur_url = "https://api.openweathermap.org/data/2.5/weather"
    fc_url  = "https://api.openweathermap.org/data/2.5/forecast"
    cur = requests.get(cur_url, params=params_base, timeout=15); cur.raise_for_status()
    fc  = requests.get(fc_url,  params=params_base, timeout=15); fc.raise_for_status()
    return cur.json(), fc.json()

# === Helper unique pour construire df_fc (prévisions 5 jours, pas 3h) — version robuste ===
@st.cache_data(ttl=900)
def build_forecast_df(fc_json):
    """
    DataFrame prévisions OWM (5 jours / pas 3h).
    Colonnes: dt, temp, feels, t_min, t_max, rain_mm, wind, desc
    """
    raw = (fc_json or {}).get("list", [])
    if not isinstance(raw, list) or len(raw) == 0:
        return pd.DataFrame(columns=["dt","temp","feels","t_min","t_max","rain_mm","wind","desc"])

    df = pd.DataFrame(raw)

    # Datetime
    df["dt"] = pd.to_datetime(df["dt"], unit="s", errors="coerce")
    df = df.dropna(subset=["dt"])

    # Helpers
    def _get_main(x, k): return x.get(k) if isinstance(x, dict) else None
    def _rain3h(v):
        if isinstance(v, dict): return float(v.get("3h", 0.0) or 0.0)
        try: return float(v)
        except: return 0.0
    def _wind(v):
        if isinstance(v, dict):
            try: return float(v.get("speed")) if v.get("speed") is not None else None
            except: return None
        try: return float(v)
        except: return None
    def _desc(lst):
        if isinstance(lst, list) and lst and isinstance(lst[0], dict):
            return str(lst[0].get("description","") or "").capitalize()
        return ""

    # Températures & ressenti
    df["temp"]  = df["main"].apply(lambda x: _get_main(x, "temp"))
    df["feels"] = df["main"].apply(lambda x: _get_main(x, "feels_like"))
    df["t_min"] = df["main"].apply(lambda x: _get_main(x, "temp_min"))
    df["t_max"] = df["main"].apply(lambda x: _get_main(x, "temp_max"))

    # Pluie + Neige éventuelle
    if "rain" not in df.columns: df["rain"] = None
    if "snow" not in df.columns: df["snow"] = None
    df["rain_mm"] = df["rain"].apply(_rain3h) + df["snow"].apply(_rain3h)

    # Vent
    if "wind" not in df.columns: df["wind"] = None
    df["wind"] = df["wind"].apply(_wind)

    # Description
    if "weather" not in df.columns: df["weather"] = None
    df["desc"] = df["weather"].apply(_desc)

    # Types
    for c in ["temp","feels","t_min","t_max","rain_mm","wind"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    cols = ["dt","temp","feels","t_min","t_max","rain_mm","wind","desc"]
    return df[cols].sort_values("dt").reset_index(drop=True)


# ------------------------------
# Scoring helpers
# ------------------------------
def _norm(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    lo, hi = s.min(), s.max()
    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        return pd.Series([0.5]*len(s), index=s.index)
    return (s - lo) / (hi - lo)

PROFILES = {
    "Général":    lambda df, tol_mm: df,
    "Plage":      lambda df, tol_mm: df[(df["avg_t_day"] >= 22) & (df["rain_sum_mm"] <= tol_mm)],
    "City-break": lambda df, tol_mm: df[(df["avg_t_day"].between(12, 26)) & (df["rain_sum_mm"] <= tol_mm)],
    "Randonnée":  lambda df, tol_mm: df[(df["avg_t_day"].between(8, 22))  & (df["rain_sum_mm"] <= tol_mm)],
    "Évasion":    lambda df, tol_mm: df[(df["avg_t_day"] >= 26) & (df["rain_sum_mm"] <= tol_mm)],
}

PROFILE_RAIN_TOL_PCT = {
    "Général":    35,
    "Plage":      25,
    "City-break": 30,
    "Randonnée":  40,
    "Évasion":    20,
}

# ==============================
# Mono-page: Météo
# ==============================
page = "Météo"

if page == "Météo":
    # Sidebar (profil & sliders)
    with st.sidebar:
        st.title("Kayak NextStop")

        # Emojis par profil
        PROFILE_EMOJI = {
            "Général": "🌍",
            "Plage": "🏖️",
            "City-break": "🏙️",
            "Randonnée": "🥾",
            "Évasion": "🔥",
        }

        # Libellés affichés = Emoji + nom + (pct %)
        def make_label(k: str) -> str:
            pct = PROFILE_RAIN_TOL_PCT.get(k, 35)
            return f"{PROFILE_EMOJI.get(k,'')} {k} ({pct} %)"

        LABELS = {k: make_label(k) for k in PROFILES.keys()}
        LABEL_TO_KEY = {v: k for k, v in LABELS.items()}

        # Options affichées avec taux par défaut
        options_labels = [LABELS[k] for k in PROFILES.keys()]

        # État courant (clé interne) + sélection affichée
        current_key = st.session_state.get("profile_key", "Général")
        st.markdown(
            "<p style='text-align: justify;'>Choisissez votre <b>profil voyageur</b> selon votre envie du moment, dans le menu déroulant ci-dessous.</p>",
            unsafe_allow_html=True
        )

        selected_label = st.selectbox(
            "**Profil voyageur**",
            options_labels,
            index=options_labels.index(LABELS[current_key]),
            key="profile_label"
        )

        profile = LABEL_TO_KEY[selected_label]  # <- clé interne propre (sans emoji)
        st.session_state["profile_key"] = profile

        with st.sidebar.expander("📅 Dates du séjour", expanded=False):
            # Sélection des dates (libellés FR)
            st.date_input("Aller",  value=st.session_state["date_in"],  key="date_in")
            st.date_input("Retour", value=st.session_state["date_out"], key="date_out")

            # Affichage des dates au format "Sam 27/09/2025"
            def _format_date_fr(d):
                jours = ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"]
                return f"{jours[d.weekday()]} {d:%d/%m/%Y}"

            st.caption(
                f"🗓️ {_format_date_fr(st.session_state['date_in'])} → {_format_date_fr(st.session_state['date_out'])}"
            )

        st.markdown("### Paramètres")
        st.markdown(
            "<p style='text-align: justify;'>Peaufinez la sélection de votre future destination en personnalisant la <b>tolérance pluie</b> et les <b>suggestions des villes</b>.</p>",
            unsafe_allow_html=True
        )


        # Valeurs par défaut dynamiques si le profil change
        default_pct_int = PROFILE_RAIN_TOL_PCT.get(profile, 35)
        if st.session_state.get("prev_prof_choice") != profile:
            st.session_state["prof_rain"] = default_pct_int   # reset slider pluie
            st.session_state["top_n"] = 5                      # reset Suggestions Top
            st.session_state["prev_prof_choice"] = profile

        # Tolérance pluie (%) avec info-bulle
        rain_tol_pct = st.slider(
            "Tolérance pluie (%)",
            min_value=0,
            max_value=100,
            value=int(st.session_state.get("prof_rain", default_pct_int)),
            step=1,
            key="prof_rain",
            help="Seuil de pluie accepté (en %) par rapport aux destinations les plus pluvieuses. Plus bas = plus ensoleillé."
        )

        # Suggestions Top des villes (ex-Top N)
        top_n = st.slider(
            "Suggestions Top des villes",
            min_value=1,
            max_value=20,
            value=int(st.session_state.get("top_n", 5)),
            step=1,
            key="top_n",
            help="Nombre de villes proposées dans la carte et le tableau."
        )

        # --- Enrichissement hébergements (OSM) à la demande ---
        st.markdown("### Hôtels")
        do_osm = st.toggle("↻ Activer la recherche pour connaître les hôtels à proximité (source OpenStreetMap)", value=False,
                           help="Active la recherche d’hébergements dans un rayon de 3 km autour du centre-ville (mairie/hôtel de ville).")


    # --- Logo + titre principal ---
        def img_b64(path): 
            return base64.b64encode(pathlib.Path(path).read_bytes()).decode()

        LOGO64 = img_b64("assets/favicon_Kayak_NextStop.png")

        st.markdown(
            f"""
            <div style="display:flex;align-items:center;justify-content:center;gap:10px;margin:6px 0 2px 0;">
              <img src="data:image/png;base64,{LOGO64}" alt="Kayak NextStop" style="height:38px;vertical-align:middle;">
              <span style="font-size:30px;font-weight:700;line-height:1;">Kayak NextStop</span>
            </div>
             """,
            unsafe_allow_html=True
        )

    
    # Mode d'emploi (H2 + colonnes)
    # --- Intro courte (mobile-first) ---
    st.subheader("🧭 Comment utiliser cette application ?")
    st.markdown(
        "Sélectionnez un **profil voyageur** et ajustez la **tolérance pluie** ainsi que le **Top des villes**. "
        "La carte affiche les meilleures destinations selon vos critères."
    )

    with st.expander("En savoir plus (profils et fonctionnement)"):
        st.markdown("""
    Chaque profil inclut une **tolérance à la pluie par défaut** (modifiable via le slider).  
    - un pourcentage bas → vous recherchez surtout du soleil ☀️  
    - un pourcentage élevé → vous êtes plus flexible et acceptez un peu de pluie 🌧️  

    **Profils proposés** (tolérance par défaut) :  
    - **🌍 Général (35 %)** : équilibre global, conditions météo variées.  
    - **🏖️ Plage (25 %)** : chaleur et faible pluie, idéal pour se détendre au soleil.  
    - **🏙️ City-break (30 %)** : températures douces et confort urbain.  
    - **🥾 Randonnée (40 %)** : climat tempéré, idéal pour marcher et explorer.  
    - **🔥 Évasion (20 %)** : chaleur intense et ambiance estivale.  

    ➡️ Vous pouvez ensuite **personnaliser** votre recherche :

    - ajuster la **Tolérance pluie (%)**,  
    - choisir le nombre de **Suggestions Top des villes**.  

    Les résultats affichent :  
    - un **Top des villes** correspondant à vos critères,  
    - une **carte interactive**,  
    - des **prévisions météo détaillées**.

    👉 Pour voir les **hôtels à proximité** dans le tableau et au survol de la carte,
    **activez l’option “Hôtels à proximité (OSM)” dans la barre latérale**.
    """)


    # Charger CSV et vues
    try:
        df_raw = load_curated_csv()
        dprof, dfw = make_views(df_raw)
    except Exception as e:
        st.error(f"Erreur de chargement des données: {e}")
        st.stop()

    # Calcul du seuil pluie en mm
    rain_max = float(dprof["rain_sum_mm"].max() or 1.0)
    tol_mm = (rain_tol_pct / 100.0) * rain_max

    # Filtrage par profil
    dsel = PROFILES[profile](dprof.copy(), tol_mm)
    if dsel.empty:
        st.info("Aucune ville ne correspond aux critères actuels. Tolérance pluie augmentée automatiquement.")
        dsel = dprof.copy()

    # Ranking
    dsel["rank_score"] = (
        0.30*_norm(dsel["avg_t_day"]) +
        0.30*(1 - _norm(dsel["rain_sum_mm"])) +
        0.20*_norm(dsel["weather_score"]) +
        0.10*_norm(dsel["hotel_score"]) +
        0.10*_norm(dsel["score_final"])
    )
    
    # --- Référentiel robuste pour la Note (/20) afin d'éviter la saturation à 20 ---
    q05 = float(dsel["score_final"].quantile(0.05))
    q95 = float(dsel["score_final"].quantile(0.95))
    denom = max(q95 - q05, 1e-9)  # évite la division par zéro

    def note20(series):
        s = pd.to_numeric(series, errors="coerce")
        return ((s - q05) / denom * 20).clip(0, 20).round(0).astype("Int64")

    sugg = dsel.sort_values("rank_score", ascending=False).head(top_n)

    # ---------------------------
    # Enrichissement OSM (optionnel via toggle)
    # ---------------------------
    if do_osm and not sugg.empty:
        with st.spinner("Recherche des hébergements à proximité (rayon 3 km)…"):
            hotels_counts = []
            hotels_mindist = []
            for _, r in sugg.iterrows():
                lat0, lon0 = float(r["lat"]), float(r["lon"])
                pts = osm_hotels_within(lat0, lon0, radius_m=3000)  # @st.cache_data dans ta fonction
                hotels_counts.append(len(pts))

                if pts:
                    # calcule la distance min en km
                    dmins = [haversine_km(lat0, lon0, la, lo) for (la, lo) in pts]
                    hotels_mindist.append(round(float(min(dmins)), 2))
                else:
                    hotels_mindist.append(None)

            sugg["Hotels_3km"] = pd.Series(hotels_counts, index=sugg.index)
            sugg["Hotel_min_dist_km"] = pd.Series(hotels_mindist, index=sugg.index)
    else:
        # Pas d’enrichissement : colonnes présentes mais vides (évite KeyError plus bas)
        if "Hotels_3km" not in sugg.columns:
            sugg["Hotels_3km"] = pd.NA
        if "Hotel_min_dist_km" not in sugg.columns:
            sugg["Hotel_min_dist_km"] = pd.NA

    # Feedback OSM (après l'enrichissement)
    if do_osm:
        if "Hotels_3km" in sugg.columns:
            col = pd.to_numeric(sugg["Hotels_3km"], errors="coerce")

            if col.isna().all():
                # On a tenté OSM mais rien n'est revenu (quota/latence)
                st.info("Hébergements non récupérés (latence/quota Overpass). Réessaie ou réactive l’option un peu plus tard.")
            elif col.sum() == 0:
                # Données présentes (pas NaN), mais vraiment 0 hôtel détecté
                st.info("Aucun hébergement détecté dans un rayon de 3 km autour des centres-villes sélectionnés.")


    # --- Top 3 recommandations (cartes décisionnelles) ---
    def _safen(x):
        try:
            return pd.to_numeric(x, errors="coerce")
        except Exception:
            return pd.NA

    def _pick_work_df():
        """
        Choisit la meilleure source disponible pour le Top 3.
        Priorité:
          1) sugg (shortlist déjà triée)
          2) dsel (ensemble filtré, prêt à être trié)
          3) dprof (agrégé par ville)
          4) autres df usuels si présents
        """
        order = ["sugg", "dsel", "dprof", "dff", "df_filtered", "df_view", "preview", "suggestions", "df_sugg"]
        for name in order:
            if name in globals():
                obj = globals()[name]
                if isinstance(obj, pd.DataFrame) and len(obj) > 0:
                    return obj
        return None


    def _score_series(df: pd.DataFrame) -> pd.Series:
        """
        Renvoie une série de score /20.
        Priorité:
          1) si 'weather_score' & 'hotel_score' existent -> pondération utilisateur
          2) sinon 'note20' si présent
          3) sinon 'meteo10' * 2 (fallback simple)
        """
        df = df.copy()
        # 1) Pondérations utilisateur si colonnes dispo
        if {"weather_score", "hotel_score"}.issubset(df.columns):
            wm = int(st.session_state.get("w_meteo", 60))
            wh = int(st.session_state.get("w_hotels", 40))
            wsum = max(wm + wh, 1)
            ws = _safen(df["weather_score"])
            hs = _safen(df["hotel_score"])
            score20 = ((wm/wsum) * ws + (wh/wsum) * hs) * 20
            return pd.to_numeric(score20, errors="coerce").clip(0, 20).round(1)

        # 2) note20 si présent
        if "note20" in df.columns:
            return pd.to_numeric(df["note20"], errors="coerce").clip(0, 20).round(1)

        # 3) fallback: meteo10 * 2
        if "meteo10" in df.columns:
            return (pd.to_numeric(df["meteo10"], errors="coerce") * 2).clip(0, 20).round(1)

        # sinon zeros (évite plantage)
        return pd.Series([pd.NA] * len(df), index=df.index)

    def render_top3_cards():
        work = _pick_work_df()
        if work is None or work.empty:
            st.info("Aucune suggestion disponible avec vos filtres.")
            return

        df_rank = work.copy()

        # Colonnes
        city_col = "city" if "city" in df_rank.columns else None
        t_col    = "avg_t_day" if "avg_t_day" in df_rank.columns else None
        rain_col = "rain_sum_mm" if "rain_sum_mm" in df_rank.columns else None
        hotel_col = "hotel_score" if "hotel_score" in df_rank.columns else None

        if city_col is None:
            st.info("Colonne 'city' manquante.")
            return

        # ----- Note globale (/20) cohérente (quantiles sur dsel si dispo, sinon df_rank) -----
        base_for_quant = df_rank
        if "dsel" in globals() and isinstance(dsel, pd.DataFrame) and not dsel.empty and "score_final" in dsel.columns:
            base_for_quant = dsel

        q05 = float(pd.to_numeric(base_for_quant["score_final"], errors="coerce").quantile(0.05))
        q95 = float(pd.to_numeric(base_for_quant["score_final"], errors="coerce").quantile(0.95))
        denom = max(q95 - q05, 1e-9)
        df_rank["__note20_int__"] = (
            (pd.to_numeric(df_rank["score_final"], errors="coerce") - q05) / denom * 20
        ).clip(0, 20).round(0).astype("Int64")

        # ----- Hôtel_/10 cohérent (percentile rank sur la même base) -----
        if hotel_col is not None:
            if "dsel" in globals() and isinstance(dsel, pd.DataFrame) and not dsel.empty and (hotel_col in dsel.columns):
                hotel_base = pd.to_numeric(dsel[hotel_col], errors="coerce")
            else:
                hotel_base = pd.to_numeric(df_rank[hotel_col], errors="coerce")

            # map des percentiles de la base vers les valeurs courantes
            hotel_pct_base = hotel_base.rank(method="average", pct=True)  # 0..1
            # on crée une série “score->pct” via quantile approché
            # (plus robuste: interpolation par rang)
            def pct_of(x):
                try:
                    # rang du point x au sein de la base (proportion val <= x)
                    return float((hotel_base <= x).mean())
                except Exception:
                    return float("nan")

            df_rank["__hotel10_int__"] = pd.to_numeric(df_rank[hotel_col], errors="coerce").map(lambda x: round(pct_of(x)*10)).astype("Int64")
        else:
            df_rank["__hotel10_int__"] = pd.NA

        # ----- Prob pluie (%) = rain_sum_mm relatif au max cohérent -----
        if rain_col is not None:
            if "dprof" in globals() and isinstance(dprof, pd.DataFrame) and not dprof.empty and "rain_sum_mm" in dprof.columns:
                max_rain = float(pd.to_numeric(dprof["rain_sum_mm"], errors="coerce").max() or 1.0)
            else:
                max_rain = float(pd.to_numeric(df_rank[rain_col], errors="coerce").max() or 1.0)

            df_rank["__prob_pct_int__"] = (
                pd.to_numeric(df_rank[rain_col], errors="coerce")
                  .div(max_rain).mul(100).clip(0, 100).round(0).astype("Int64")
            )
        else:
            df_rank["__prob_pct_int__"] = pd.NA

        # ----- Tri: Note -> Hôtel -> T° (desc) -----
        sort_keys = ["__note20_int__", "__hotel10_int__"]
        asc = [False, False]
        if t_col:
            sort_keys.append(t_col)
            asc.append(False)

        top = df_rank.sort_values(sort_keys, ascending=asc, kind="mergesort").head(3)


        # ----- Top 3 Recommandations rapides -----
        st.subheader("Top 3 — recommandations rapides")

        # --- Styles: vertical (desktop) + horizontal (mobile) ---
        st.markdown(
            """
            <style>
              /* Mobile: on masque le trait vertical et on ajoute une ligne sous chaque carte */
              @media (max-width: 640px){
                .vsep { display: none !important; }
                .kcard { border-bottom: 1px solid #D1D5DB; margin-bottom: 6px; }
              }
            </style>
            """,
            unsafe_allow_html=True
        )

        # --- Colonnes avec colonnes-séparateurs (min-height pour rendre le trait visible) ---
        n = len(top)
        if n == 1:
            cols = st.columns([1])
        elif n == 2:
            cols = st.columns([1, 0.03, 1])                 # c1 | sep | c2
        else:  # n >= 3
            cols = st.columns([1, 0.03, 1, 0.03, 1])        # c1 | sep | c2 | sep | c3

        MIN_H = 220  # hauteur mini du séparateur en px (à ajuster si besoin)

        def _render_card(col, row):
            with col:
                st.markdown("<div class='kcard' style='padding:8px 10px 0;'>", unsafe_allow_html=True)

                city = str(row[city_col])
                t    = float(row[t_col]) if t_col and pd.notna(row[t_col]) else None
                prob = int(row["__prob_pct_int__"]) if pd.notna(row.get("__prob_pct_int__", None)) else None
                note = int(row["__note20_int__"])  if pd.notna(row.get("__note20_int__", None))  else None
                h10  = int(row["__hotel10_int__"]) if pd.notna(row.get("__hotel10_int__", None)) else None

                st.markdown(f"**{city}**")
                a, b = st.columns(2)
                a.metric("T° moy.", f"{t:.1f} °C" if t is not None else "—")
                b.metric("Prob pluie (%)", f"{prob} %" if prob is not None else "—")
                c, d = st.columns(2)
                c.metric("Note globale (/20)", f"{note}" if note is not None else "—")
                d.metric("Hôtel_/10", f"{h10}" if h10 is not None else "—")

                try:
                    links = make_hotel_links(city, st.session_state["date_in"], st.session_state["date_out"])
                    st.markdown(f"[Voir sur Kayak]({links['Kayak']})")
                    st.caption(f"Alternatives : [Google Hotels]({links['Google Hotels']}) · [Booking]({links['Booking']})")
                except Exception:     
                    st.caption("👉 Renseignez les dates dans la sidebar pour activer le lien.")

                st.markdown("</div>", unsafe_allow_html=True)

        def _render_sep(col):     
            with col:
                # Trait vertical visible grâce à une hauteur minimale
                st.markdown(
                    f"<div class='vsep' style='width:2px; min-height:{MIN_H}px; background:#D1D5DB; margin:0 auto;'></div>",
                    unsafe_allow_html=True
                )

        rows = list(top.iterrows())
        if n == 1:    
            _render_card(cols[0], rows[0][1])
        elif n == 2:      
            _render_card(cols[0], rows[0][1])
            _render_sep(cols[1])
            _render_card(cols[2], rows[1][1])
        else:     
            _render_card(cols[0], rows[0][1])
            _render_sep(cols[1])
            _render_card(cols[2], rows[1][1])
            _render_sep(cols[3])
            _render_card(cols[4], rows[2][1])
        

    # -- rendu
    render_top3_cards()

    
    # Carte France / Monde (Top N)
    st.subheader("🗺️ Carte — Top des villes selon le profil")
    df_map = sugg.copy().reset_index(drop=True)
    if df_map.empty:
        st.info("Aucune ville à afficher sur la carte pour la sélection actuelle.")
    else:
        df_map.insert(0, "Top", range(1, len(df_map) + 1))
        df_map["Top_str"] = df_map["Top"].astype(str)
        df_map["lat"] = pd.to_numeric(df_map["lat"], errors="coerce")
        df_map["lon"] = pd.to_numeric(df_map["lon"], errors="coerce")
        df_map = df_map.dropna(subset=["lat","lon"]).reset_index(drop=True)

        FR_BBOX = {"lat_min": 41.0, "lat_max": 51.7, "lon_min": -5.5, "lon_max": 9.9}
        df_fr = df_map[
            df_map["lat"].between(FR_BBOX["lat_min"], FR_BBOX["lat_max"]) &
            df_map["lon"].between(FR_BBOX["lon_min"], FR_BBOX["lon_max"])
        ].copy()
        show = df_fr if not df_fr.empty else df_map

        hotel_raw = pd.to_numeric(show.get("hotel_score", pd.Series([None]*len(show))), errors="coerce")
        hotel_pct = hotel_raw.rank(method="average", pct=True)
        show["size_bubble"] = (hotel_pct * 22).clip(6, 22)

        is_fr_view = (show is df_fr)
        center = {"lat": 46.5, "lon": 2.5} if is_fr_view else {"lat": float(show["lat"].mean()), "lon": float(show["lon"].mean())}
        zoom = 4.5 if is_fr_view else 2.3

        # Colonnes pour tooltip
        show["note20_int"] = note20(show["score_final"])
        show["meteo10_int"] = (pd.to_numeric(show["weather_score"], errors="coerce") * 10).clip(0, 10).round(0).astype("Int64")
        rain_max_safe = float(dprof["rain_sum_mm"].max() or 1.0)
        show["prob_pct_int"] = (
            pd.to_numeric(show["rain_sum_mm"], errors="coerce")
              .div(rain_max_safe).mul(100).clip(0, 100).round(0).astype("Int64")
        )
        if "Hotels_3km" not in show.columns:
            show["Hotels_3km"] = 0
        if "Hotel_min_dist_km" not in show.columns:
            show["Hotel_min_dist_km"] = None
        show["hotel_10"] = (hotel_pct * 10).round(1)

        # Carte Plotly / OpenStreetMap — couleur = Prob pluie (%)
        fig_fr = px.scatter_mapbox(
            show,
            lat="lat", lon="lon",
            text="Top_str",
            hover_name="city",
            size="size_bubble",              # taille = hôtels (inchangée)
            color="prob_pct_int",            # couleur = probabilité pluie (%)
            color_continuous_scale=[
                [0.0, "#FFD54F"],            # jaune clair (0%)
                [1.0, "#1E88E5"]             # bleu soutenu (100%)
            ],
            range_color=[0, 100],
            zoom=zoom, center=center,
            height=MAP_HEIGHT,
            title=f"Top {len(show)} — Profil {profile}" + (" (France)" if is_fr_view else "")
        )

        fig_fr.update_coloraxes(colorbar_title="Prob pluie (%)")

        # Mini-légende overlay (discrète, mobile-friendly)
        fig_fr.add_annotation(
            xref="paper", yref="paper",
            x=0.01, y=0.99,                  # coin haut-gauche
            align="left", showarrow=False,
            bordercolor="rgba(0,0,0,0.2)", borderwidth=1, borderpad=6,
            bgcolor="rgba(255,255,255,0.85)",
            font=dict(size=12),
            text=(
                "<b>ℹ️ Légende</b><br>"
                "• <b>Taille</b> : qualité hôtels (OSM)<br>"
                "• <b>Couleur</b> : prob. pluie (%)<br>"
                "  jaune = sec → bleu = pluvieux"
            )
        )

        fig_fr.update_traces(
            marker=dict(opacity=0.9),
            textposition="top center",
            customdata=show[[
                "Top", "note20_int", "meteo10_int", "avg_t_day", "prob_pct_int",
                "hotel_10", "Hotels_3km", "Hotel_min_dist_km"
            ]],
            hovertemplate=(
                "<b>Ville: %{hovertext}</b><br>"
                "Top %{customdata[0]}<br>"
                "Note globale (/20): %{customdata[1]}<br>"
                "Météo (/10): %{customdata[2]}<br>"
                "Température: %{customdata[3]:.1f} °C<br>"
                "Prob pluie: %{customdata[4]}%<br>"
                "Hôtel_/10: %{customdata[5]}<br>"
                "Hôtels (≤3 km): %{customdata[6]}<br>"
                "Hôtel le + proche: %{customdata[7]} km<extra></extra>"
            )
        )
        fig_fr.update_layout(mapbox_style="open-street-map", margin=dict(l=0, r=0, t=50, b=0))
        
        st.plotly_chart(
            fig_fr,
            use_container_width=True,
            config={
                "scrollZoom": True,
                "displayModeBar": True,
                "modeBarButtonsToAdd": ["zoomInMapbox", "zoomOutMapbox", "resetViewMapbox"]
            },
            key=f"map_{profile}_{top_n}_{int(rain_tol_pct)}"
        )

        # ----- 🔎 Pourquoi le Top 3 peut différer du Top 5 ? -----
        with st.expander("🔎 Pourquoi le Top 3 peut différer du Top 5 ?", expanded=False):
            st.markdown("""
            Le **Top 3** (cartes rapides en haut) et le **Top 5** (carte et tableau) ne répondent pas exactement à la même question :

            - **Top 5 (carte/tableau)** : on classe **toutes les villes filtrées** et on affiche les 5 meilleures.
            - **Top 3 (cartes rapides)** : on prend une **sélection réduite** et on met en avant seulement 3 villes.

            Cela peut créer des différences, pour plusieurs raisons :
            - pas toujours la même base de calcul (ensemble complet vs. sous-ensemble),
            - méthodes de calcul des notes différentes (normalisation, pourcentages),
            - arrondis ou égalités qui modifient légèrement l’ordre,
            - et bien sûr : on n’affiche pas le même nombre de résultats.

            👉 En résumé : les deux listes se complètent.  
            Le **Top 3** donne une vue rapide et compacte, tandis que le **Top 5** offre plus de détails pour comparer.
            """)    

        # Tableau Suggestions (placer APRES la carte)
    with st.expander("✅ Suggestions de villes (selon le profil)", expanded=False):
        _s = sugg.copy().reset_index(drop=True)

        # Rang Top
        _s.insert(0, "Top", range(1, len(_s) + 1))

        # Colonnes calculées / normalisées pour l'affichage
        _s["Note globale (/20)"] = note20(_s["score_final"])
        _s["Météo_/10"] = (
            pd.to_numeric(_s["weather_score"], errors="coerce").mul(10).clip(0, 10).round(0).astype("Int64")
        )
        _s["Temp (°C)"] = pd.to_numeric(_s["avg_t_day"], errors="coerce").round(1)

        rain_max_safe = float(dprof["rain_sum_mm"].max() or 1.0)
        _s["Prob pluie (%)"] = (
            pd.to_numeric(_s["rain_sum_mm"], errors="coerce")
              .div(rain_max_safe).mul(100).clip(0, 100).round(0).astype("Int64")
        )

        # Hôtel_/10 en relatif (plus discriminant)
        hotel = pd.to_numeric(_s["hotel_score"], errors="coerce")
        hotel_pct = hotel.rank(method="average", pct=True)  # 0..1
        _s["Hôtel_/10"] = (hotel_pct * 10).round(1)

        # Colonnes OSM si présentes
        if "Hotels_3km" in _s.columns:
            _s["Hôtels (≤3 km)"] = _s["Hotels_3km"].astype("Int64")
        if "Hotel_min_dist_km" in _s.columns:
            _s["Hôtel le + proche (km)"] = _s["Hotel_min_dist_km"]

        # Ordre final des colonnes
        base_cols = ["Top", "city", "Note globale (/20)", "Météo_/10", "Temp (°C)", "Prob pluie (%)", "Hôtel_/10"]
        extra_cols = [c for c in ["Hôtels (≤3 km)", "Hôtel le + proche (km)"] if c in _s.columns]
        out = _s[base_cols + extra_cols].rename(columns={"city": "Villes"})

        st.dataframe(out.reset_index(drop=True), use_container_width=True)


    # ----- ℹ️ Légende détaillée — comment lire la carte et le tableau ? -----
    with st.expander("ℹ️ Légende détaillée — comment lire la carte et le tableau ?"):
        st.markdown("""
    ### Ce que représente la **carte**
    - **Taille du point** : reflète la **qualité de l’offre hôtelière** proche du **centre de la ville** (repère = mairie/hôtel de ville), dans un **rayon de 3 km**.  
      *Source : OpenStreetMap (données ouvertes).*  
    - **Couleur du point** : indique la **probabilité de pluie (%)** sur la période observée (**5 jours**).  
      **Jaune = plutôt sec**, **bleu = plutôt pluvieux**.  
      *Source : OpenWeatherMap (prévisions météo).*

    ### Comment lire la carte et le tableau des villes
    - **Top** : la position de la ville dans vos suggestions (1 = meilleure).  
    - **Note globale (/20)** : une **note comparative** entre les villes proposées, calculée selon votre **profil voyageur**.  
      Elle prend en compte différents critères concrets comme la **probabilité de pluie**, la **température moyenne**, les **conditions météo**, le **vent** (selon le profil) et la **qualité hôtelière**.  

      À chaque **changement de profil**, les notes sont recalculées :  
      - la meilleure ville de votre sélection obtient **20/20**,  
      - les moins favorables se rapprochent de **0/20**,  
      - les autres se répartissent entre les deux.  

      👉 Cela évite d’avoir “20/20 partout” et rend le classement beaucoup plus clair et utile pour comparer les villes.

    - **Météo (/10)** : un **indice météo** sur 10, calculé en fonction de votre **profil voyageur**.  
      Plus la valeur est élevée, plus les conditions correspondent aux critères de votre profil (par exemple : soleil et chaleur pour “Plage”, douceur et peu de pluie pour “City-break”).
    - **Température (°C)** : **moyenne** prévue sur les 5 prochains jours.  
    - **Prob pluie (%)** : indique la **probabilité moyenne de pluie** sur la période des 5 prochains jours.  
      Plus le pourcentage est élevé, plus les averses sont fréquentes dans les prévisions.  
      Plus il est bas, plus la période est ensoleillée.
    - **Hôtel_/10** : reflète la **présence et la proximité d’hébergements** autour de la ville (rayon de 3 km autour de la mairie/hôtel de ville).  
      Plus la valeur est élevée, plus l’offre hôtelière est dense et proche.  
      ⚠️ Ce n’est pas une note issue d’avis clients (comme TripAdvisor ou Google), mais un indicateur basé sur les données ouvertes d’**OpenStreetMap**.
  
        - **Hôtels (≤ 3 km)** : **nombre d’hébergements** détectés dans un rayon de 3 km autour de la **mairie/hôtel de ville**.  
        - **Hôtel le + proche (km)** : distance au **premier hébergement** de la **mairie/hôtel de ville**.  
          *Données hébergements : source OpenStreetMap.*

        **Sources**  
        - Météo & prévisions : *OpenWeatherMap* (5 jours, pas de 3 h).  
        - Hébergements : *OpenStreetMap* (données ouvertes, saisies collaboratives).  
        """)
    
    # --- Scatterplot Note/Température/Pluie ---
    with st.expander("📊 Analyse croisée : Score vs Température et Pluie", expanded=False):
        try:
            df_scatter = sugg.copy()

            # Normalisation pour cohérence
            df_scatter["Note globale (/20)"] = note20(df_scatter["score_final"])
            df_scatter["Température (°C)"] = pd.to_numeric(df_scatter["avg_t_day"], errors="coerce").round(1)
            rain_max_safe = float(dprof["rain_sum_mm"].max() or 1.0)
            df_scatter["Prob pluie (%)"] = (
                pd.to_numeric(df_scatter["rain_sum_mm"], errors="coerce")
                  .div(rain_max_safe).mul(100).clip(0, 100).round(0).astype("Int64")
            )

            fig_scatter = px.scatter(
                df_scatter,
                x="Température (°C)",
                y="Note globale (/20)",
                color="Prob pluie (%)",
                color_continuous_scale="Blues",
                hover_name="city",
                text="city",  # <- affiche le nom de la ville
                labels={
                    "city": "Ville",
                    "Température (°C)": "Température (°C)",
                    "Note globale (/20)": "Note globale (/20)",
                    "Prob pluie (%)": "Probabilité pluie (%)"
                },
                title="Note globale (/20) en fonction de la Température et de la Probabilité de pluie"
            )

            # Position du texte (au-dessus du point)
            fig_scatter.update_traces(
                textposition="top center",
                marker=dict(size=14, line=dict(width=0)),
                hovertemplate=(
                    "Ville: %{hovertext}<br>"
                    "Note globale (/20): %{y:.0f}<br>"
                    "Température: %{x:.1f} °C<br>"
                    "Prob pluie: %{marker.color:.0f}%<extra></extra>"
                )
            )

            # Points principaux (taille + tooltip personnalisé)
            fig_scatter.update_traces(
                marker=dict(size=14, line=dict(width=0)),
                hovertemplate=(
                    "Ville: %{hovertext}<br>"
                    "Note globale (/20): %{y:.0f}<br>"
                    "Température: %{x:.1f} °C<br>"
                    "Prob pluie: %{marker.color:.0f}%<extra></extra>"
                )
            )

            # Ajout du contour noir pour Prob pluie < 25 %
            low_rain = df_scatter[df_scatter["Prob pluie (%)"] < 25]
            if not low_rain.empty:
                fig_scatter.add_scatter(
                    x=low_rain["Température (°C)"],
                    y=low_rain["Note globale (/20)"],
                    mode="markers",
                    marker=dict(
                        size=14,
                        color=low_rain["Prob pluie (%)"],
                        colorscale="Blues",
                        line=dict(width=1.5, color="black")
                    ),
                    hovertext=low_rain["city"],
                    hovertemplate=(
                        "Ville: %{hovertext}<br>"
                        "Note globale (/20): %{y:.0f}<br>"
                        "Température: %{x:.1f} °C<br>"
                        "Prob pluie: %{marker.color:.0f}%<extra></extra>"
                    ),
                    showlegend=False
                )


            fig_scatter.update_layout(
                margin=dict(l=10, r=10, t=60, b=10),
                height=500,
                plot_bgcolor="#F5F5F5",   # fond du graphe gris clair
                paper_bgcolor="#F5F5F5"   # fond global identique
            )

        
            st.plotly_chart(fig_scatter, use_container_width=True, key="scatter_note_temp_rain")
        except Exception as e:
            st.info(f"Graphe non disponible : {e}")


    # Sélection finale de la ville pour la météo détaillée
    all_cities = dfw["city"].tolist()
    pref_list = sugg["city"].tolist()
    options = (["— Villes suggérées —"] + pref_list + ["— Toutes les villes —"] + all_cities)
    pick = st.selectbox("Ville", options, index=1 if pref_list else 0, key="city_pick")
    if pick in ("— Villes suggérées —", "— Toutes les villes —"):
        city = (pref_list[0] if pref_list else all_cities[0])
    else:
        city = pick
    sel = dfw[dfw["city"] == city]
    if sel.empty:
        city = all_cities[0]
        sel = dfw[dfw["city"] == city]
    row = sel.iloc[0]
    lat, lon = float(row["lat"]), float(row["lon"])
    st.subheader(f"📍 Météo actuelle à {city}")

    # Appel OWM
    OWM_KEY = get_owm_key()
    if not OWM_KEY:
        st.error("Clé OpenWeatherMap manquante. Ajoute-la dans st.secrets['OPENWEATHER_API_KEY'] ou variable d'environnement OPENWEATHER_API_KEY.")
        st.stop()

    try:
        now, fc = fetch_weather(lat, lon, OWM_KEY, units="metric")
        # Construction unique de df_fc (réutilisée partout)
        df_fc = build_forecast_df(fc)
        if df_fc.empty:
            st.warning("⚠️ Aucune donnée météo reçue de l’API OpenWeatherMap.")

        # décalage horaire local OWM
        tz_offset_sec = int((fc or {}).get("city", {}).get("timezone", 0))
        tz_delta = pd.Timedelta(seconds=tz_offset_sec)
        
        # Datetime locale (naïve) = UTC + offset ville
        df_fc["dt_local"] = df_fc["dt"] + tz_delta


    except Exception as e:
        st.error(f"Erreur API OpenWeatherMap : {e}")
        st.stop()

    # === Cartes "Météo actuelle" (4 tuiles uniformes) ===
    try:
        col1, col2, col3, col4 = st.columns(4)

        # 🌡️ Température + ressenti
        temp = now["main"]["temp"]
        feels = now["main"].get("feels_like", None)
        with col1:  
            st.metric("🌡️ Temp actuelle", f"{temp:.1f} °C")
            if feels is not None:
                st.caption(f"Ressenti : {feels:.1f} °C")

        # 💧 Humidité
        with col2:  
            st.metric("💧 Humidité", f"{now['main']['humidity']} %")

        # 🌬️ Vent
        wind = now.get("wind", {}).get("speed", None)
        with col3:
            if wind is not None:
                st.metric("🌬️ Vent", f"{wind:.1f} m/s")
            else:
                st.metric("🌬️ Vent", "—")

        # 📝 Conditions
        desc = (now.get("weather", [{}])[0].get("description", "") or "—").capitalize()
        with col4:
            st.metric("📝 Conditions", desc)
    except Exception:
        pass


    # === Prévisions 24h (table) ===
    with st.expander(f"⏱️ Prévisions météo (24 heures) à {city}", expanded=False):

        try:
            # Prochaines 24h → 8 pas (3h)
            # Référence temps locale (naïve) pour filtrer les prochaines 24h
            now_local = pd.Timestamp.utcnow().tz_localize(None) + tz_delta
            next_24h = df_fc[df_fc["dt_local"] > now_local].head(8).copy()

            tbl24 = next_24h[["dt_local","desc","temp","feels","rain_mm","wind"]].rename(columns={
                "dt_local": "Heure",
                "desc": "Conditions",
                "temp": "Temp (°C)",
                "feels": "Ressenti (°C)",
                "rain_mm": "Pluie (mm/3h)",
                "wind": "Vent (m/s)"
            })

            # Format FR pour l’heure (abréviations FR)
            abbr_map = {"Mon":"Lun","Tue":"Mar","Wed":"Mer","Thu":"Jeu","Fri":"Ven","Sat":"Sam","Sun":"Dim"}
            tmp = tbl24["Heure"].dt.strftime("%a %H:%M")
            tbl24["Heure"] = tmp.apply(lambda s: f"{abbr_map.get(s.split()[0], s.split()[0])} {s.split()[1]}")

            st.dataframe(tbl24.reset_index(drop=True), use_container_width=True)

        except Exception as e:
            st.info(f"Tableau 24h non disponible : {e}")

    # === Titre 5 jours (les graphes existants viennent ensuite) ===
    with st.expander(f"📅 Prévisions météo (5 jours) à {city}", expanded=False):

        # === Prévisions (courbes & barres) — 5 jours ===
        try:
            # Lignes verticales à minuit (une par jour)
            day_boundaries = sorted(df_fc["dt"].dt.normalize().unique())
            shapes_midnight = [
                dict(type="line",
                     x0=pd.Timestamp(d), x1=pd.Timestamp(d),
                     y0=0, y1=1, yref="paper",
                     line=dict(color="rgba(0,0,0,0.12)", width=1, dash="dot"))
                for d in day_boundaries
            ]

        # === Graphe combiné Température + Pluie (5 jours) ===
            

            # Repères à minuit LOCAL
            day_boundaries = sorted(df_fc["dt_local"].dt.normalize().unique())
            shapes_midnight = [
                dict(type="line", x0=pd.Timestamp(d), x1=pd.Timestamp(d),
                     y0=0, y1=1, yref="paper",
                     line=dict(color="rgba(0,0,0,0.12)", width=1, dash="dot"))
                for d in day_boundaries
            ]

            fig_tr = make_subplots(specs=[[{"secondary_y": True}]])

            # Barres pluie (axe gauche)
            fig_tr.add_bar(
                x=df_fc["dt_local"], y=df_fc["rain_mm"],
                name="Pluie (mm/3h)",
                opacity=0.8,
                hovertemplate="%{x|%d/%m %H:%M}<br>Pluie: %{y:.1f} mm/3h<extra></extra>"
            )

            # Courbe température (axe droit)
            fig_tr.add_scatter(
                x=df_fc["dt_local"], y=df_fc["temp"],
                mode="lines", name="Température (°C)",
                line=dict(color="#FFA84D", width=3),
                hovertemplate="%{x|%d/%m %H:%M}<br>Température: %{y:.1f} °C<extra></extra>"
            )

            fig_tr.update_layout(
                title="Évolution des températures & précipitations (5 jours)",
                hovermode="x unified",
                shapes=shapes_midnight,
                margin=dict(l=10, r=10, t=60, b=10),
                plot_bgcolor="#FFFFFF",
                paper_bgcolor="#FFFFFF",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            fig_tr.update_layout(height=360)

            fig_tr.update_xaxes(title_text="Date/Heure (local)", tickformat="%d/%m %H:%M")
            fig_tr.update_yaxes(title_text="Pluie (mm/3h)", secondary_y=False)
            fig_tr.update_yaxes(title_text="Température (°C)", secondary_y=True)
    
            st.plotly_chart(fig_tr, use_container_width=True, key="meteo_temp_rain_combo")
    
        except Exception as e:
            st.info(f"Prévisions non affichées : {e}")    

    # -------- Vent (ligne BLEU CLAIR) --------
    with st.expander("🌬️ Vent prévu (optionnel)"):
        # --- Unité du vent (km/h par défaut, mémorisée en session) ---
        opts = ["m/s", "km/h"]
        default_index = 1 if st.session_state.get("wind_unit", "km/h") == "km/h" else 0
        wind_unit = st.radio("Unité du vent", opts, index=default_index, horizontal=True, key="wind_unit")

        # Compatibilité si le reste du code utilise encore 'unit'
        unit = wind_unit

        df_fc_wind = df_fc.copy()
        if unit == "km/h":
            df_fc_wind["wind_disp"] = df_fc_wind["wind"] * 3.6
            y_label = "Vent (km/h)"
            hover_tpl = "Heure: %{x|%d/%m %H:%M}<br>Vent: %{y:.1f} km/h<extra></extra>"
        else:
            df_fc_wind["wind_disp"] = df_fc_wind["wind"]
            y_label = "Vent (m/s)"
            hover_tpl = "Heure: %{x|%d/%m %H:%M}<br>Vent: %{y:.1f} m/s<extra></extra>"

        # df_fc_wind = df_fc.copy() déjà présent
        fig_w = px.line(
            df_fc_wind,
            x="dt_local", y="wind_disp",
            labels={"dt_local": "Date/Heure (local)", "wind_disp": y_label},
            title="Vent prévu (5 jours)"
        )

        fig_w.update_traces(
            line=dict(color="#5DADE2", width=3),
            hovertemplate=hover_tpl
        )
        fig_w.update_layout(
            hovermode="x unified",
            shapes=shapes_midnight,
            margin=dict(l=10, r=10, t=60, b=10),
            height=320
        )

        st.plotly_chart(fig_w, use_container_width=True, key="meteo_wind_ms")

