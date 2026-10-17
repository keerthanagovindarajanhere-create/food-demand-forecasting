"""Shared Streamlit helpers for FOODFLOW: cached loaders, styling, KPI cards."""

from __future__ import annotations

import pandas as pd
import streamlit as st

import foodflow_core as core

# Colours used consistently across every chart.
ACTUAL = "#5b6770"      # observed demand
REFERENCE = "#1f7a5c"   # reference XGBoost
TAIL = "#d9822b"        # tail-weighted XGBoost / forecasts
BASELINE = "#b8c2cc"
PLOT_TEMPLATE = "simple_white"


# ------------------------------------------------------------
# Cached loaders (loaded once per server process)
# ------------------------------------------------------------

@st.cache_resource(show_spinner="Loading frozen XGBoost model...")
def model():
    return core.load_model()


@st.cache_data
def feature_columns():
    return core.load_feature_columns()


@st.cache_data(show_spinner="Loading demand history...")
def history():
    return core.load_history()


@st.cache_data(show_spinner="Loading validation results...")
def validation():
    v = core.load_validation_errors()
    t = core.load_tail_predictions()
    v["tail_predicted"] = t["predicted"].values
    v["demand_regime"] = t["demand_regime"].values
    return v


@st.cache_data
def csv(path_str: str) -> pd.DataFrame:
    return pd.read_csv(path_str)


@st.cache_data
def headline_metrics() -> dict:
    v = validation()
    base = csv(str(core.BASELINE_PATH))
    curve = core.cumulative_squared_error_share(v["sales"], v["predicted"])
    return {
        "xgb_mae": core.mae(v["sales"], v["predicted"]),
        "xgb_rmse": core.rmse(v["sales"], v["predicted"]),
        "xgb_rmsle": core.rmsle(v["sales"], v["predicted"]),
        "tail_mae": core.mae(v["sales"], v["tail_predicted"]),
        "tail_rmse": core.rmse(v["sales"], v["tail_predicted"]),
        "n_val": len(v),
        "baselines": base,
        "top1": core.share_at(curve, 1),
        "top5": core.share_at(curve, 5),
    }


@st.cache_data
def series_validation(store_nbr: int, family: str) -> pd.DataFrame:
    v = validation()
    return v[(v["store_nbr"] == store_nbr) & (v["family"] == family)].sort_values("date")


# ------------------------------------------------------------
# Presentation
# ------------------------------------------------------------

CSS = """
<style>
.block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1250px;}
.ff-card {
  border: 1px solid rgba(128,128,128,0.25); border-radius: 12px;
  padding: 14px 18px; background: rgba(31,122,92,0.05); min-height: 112px;
}
.ff-card .label {font-size: 0.82rem; opacity: 0.75; margin-bottom: 2px;}
.ff-card .value {font-size: 1.65rem; font-weight: 700; line-height: 1.2;}
.ff-card .note {font-size: 0.78rem; opacity: 0.7; margin-top: 4px;}
.ff-hero {
  border-radius: 14px; padding: 22px 26px; margin: 6px 0 14px 0;
  background: linear-gradient(135deg, rgba(31,122,92,0.12), rgba(217,130,43,0.10));
  border: 1px solid rgba(128,128,128,0.2);
}
.ff-hero .big {font-size: 2.6rem; font-weight: 800; line-height: 1.1;}
.ff-tag {display:inline-block; font-size:0.75rem; padding:2px 8px; border-radius:999px;
  background: rgba(217,130,43,0.18); margin-left: 6px; vertical-align: middle;}
</style>
"""


def setup_page():
    st.markdown(CSS, unsafe_allow_html=True)


def kpi(col, label: str, value: str, note: str = ""):
    col.markdown(
        f'<div class="ff-card"><div class="label">{label}</div>'
        f'<div class="value">{value}</div>'
        f'<div class="note">{note}</div></div>',
        unsafe_allow_html=True,
    )


def style_fig(fig, height: int = 380):
    fig.update_layout(
        template=PLOT_TEMPLATE,
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified" if fig.layout.hovermode is None else fig.layout.hovermode,
    )
    if fig.layout.title.text:  # keep the title above the legend instead of on top of it
        fig.update_layout(margin=dict(t=85),
                          title=dict(y=0.98, yanchor="top", x=0, xanchor="left",
                                     font=dict(size=15)))
    return fig


def safe_load() -> bool:
    """Load model + history once; show a clear error instead of a traceback."""
    try:
        model()
        feature_columns()
        history()
        validation()
        return True
    except FileNotFoundError as e:
        st.error(f"A required project file is missing: `{e.filename}`")
        return False
    except Exception as e:  # pragma: no cover - defensive UI path
        st.error("FOODFLOW could not load its model or research artifacts.")
        st.exception(e)
        return False


def series_picker(key_prefix: str, default_store: int = 44,
                  default_family: str = "BEVERAGES"):
    """Store + family selectors shared by Forecast and Planner pages."""
    hist, _ = history()
    stores = sorted(hist["store_nbr"].unique().tolist())
    families = sorted(hist["family"].unique().tolist())
    if f"{key_prefix}_store" not in st.session_state:
        st.session_state[f"{key_prefix}_store"] = st.session_state.get("sel_store", default_store)
        st.session_state[f"{key_prefix}_family"] = st.session_state.get("sel_family", default_family)
    c1, c2 = st.columns(2)
    store = c1.selectbox("Store", stores, key=f"{key_prefix}_store")
    family = c2.selectbox("Product family", families, key=f"{key_prefix}_family")
    st.session_state["sel_store"], st.session_state["sel_family"] = store, family
    return int(store), family
