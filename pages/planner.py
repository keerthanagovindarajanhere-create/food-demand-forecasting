import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import foodflow_core as core
import ui

st.title("🧑‍🍳 Smart Preparation Planner")
st.write(
    "Turns a FOODFLOW forecast into an **illustrative** preparation quantity and compares "
    "options with a simple shortage-versus-waste simulation."
)
st.caption(
    "⚠️ Simulation only. Costs are your assumptions, and 'demand scenarios' are built from this "
    "model's own past errors. No real inventory, savings or waste reduction is measured or claimed."
)

model, fcols = ui.model(), ui.feature_columns()
hist, _ = ui.history()

# ------------------------------------------------------------ Forecast input
with st.container(border=True):
    store, family = ui.series_picker("pl")
    c3, c4 = st.columns(2)
    first = core.LAST_OBSERVED_DATE + pd.Timedelta(days=1)
    target = c3.date_input("Preparation date", value=first.date(), min_value=first.date(),
                           max_value=(core.LAST_OBSERVED_DATE
                                      + pd.Timedelta(days=core.MAX_HORIZON_DAYS)).date())
    promo = c4.number_input("Items on promotion", min_value=0, max_value=800, value=0, step=1)

horizon = (pd.Timestamp(target) - core.LAST_OBSERVED_DATE).days
series = core.series_history(hist, store, family)
try:
    forecast = float(core.forecast_series(model, fcols, series, store, family, horizon,
                                          promo)["prediction"].iloc[-1])
except ValueError as e:
    st.error(f"Forecast not generated: {e}")
    st.stop()

val = ui.series_validation(store, family)
residuals = (val["sales"] - val["predicted"]).to_numpy()  # actual - predicted, 28 days

if forecast < 1:
    st.info(
        f"The forecast for {family} at store {store} is {forecast:.2f} units — this series has "
        "(almost) no recent sales, so there is nothing meaningful to plan. Try another store or "
        "family, e.g. BREAD/BAKERY, DAIRY or PRODUCE."
    )
    st.stop()

if horizon > 1:
    st.warning("This date uses a recursive (multi-day) forecast, so real uncertainty is larger "
               "than the simulated scenarios below suggest.")

# ------------------------------------------------------------ Assumptions
st.subheader("1 · Assumptions")
a1, a2, a3 = st.columns(3)
buffer = a1.slider("Safety buffer on top of the forecast (%)", 0, 50, 10, 1)
cu = a2.number_input("Cost of one unit short (lost margin / goodwill)", 0.1, 1000.0, 3.0, 0.5,
                     help="Illustrative value in any currency per unit of unmet demand.")
co = a3.number_input("Cost of one unit wasted (unsold / spoiled)", 0.1, 1000.0, 1.0, 0.5,
                     help="Illustrative value in the same currency per unit prepared but not sold.")

prep = core.buffered_quantity(forecast, buffer)
cr = core.critical_ratio(cu, co)
opt = core.optimal_quantity(forecast, residuals, cu, co)
res_buffer = core.simulate_outcomes(prep, forecast, residuals, cu, co)
res_opt = core.simulate_outcomes(opt, forecast, residuals, cu, co)
res_fc = core.simulate_outcomes(forecast, forecast, residuals, cu, co)

# ------------------------------------------------------------ Recommendation
st.subheader("2 · Preparation quantity")
k = st.columns(4)
ui.kpi(k[0], "Forecast demand", f"{forecast:,.0f}", f"{family}, store {store}, {pd.Timestamp(target):%d %b}")
ui.kpi(k[1], f"With {buffer}% buffer", f"{np.ceil(prep):,.0f}",
       f"simulated stock-out chance {res_buffer['stockout_prob'] * 100:.0f}%")
ui.kpi(k[2], "Cost-balanced quantity", f"{np.ceil(opt):,.0f}",
       f"critical ratio {cr:.2f} → {(opt / forecast - 1) * 100:+.1f}% vs forecast")
ui.kpi(k[3], "Model error scenarios", f"{len(residuals)}",
       f"validation days for this series (MAE {np.abs(residuals).mean():,.0f})")

st.write("")
with st.expander("How these numbers are calculated", expanded=False):
    st.markdown(
        r"""
- **Buffered quantity** $= \text{forecast} \times (1 + \text{buffer}\%)$.
- **Demand scenarios.** For this store-family the frozen model made 28 validation forecasts.
  Each past error $r_i = \text{actual}_i - \text{predicted}_i$ becomes one scenario:
  $D_i = \max(0,\ \text{forecast} + r_i)$.
- **Cost of preparing $Q$** in a scenario
  $= C_u \cdot \max(D_i - Q, 0) + C_o \cdot \max(Q - D_i, 0)$, averaged over all scenarios.
- **Cost-balanced quantity** (newsvendor rule): the quantile of the $D_i$ at the
  *critical ratio* $C_u / (C_u + C_o)$. If a shortage costs 3× more than waste, the critical
  ratio is 0.75, so you prepare enough to cover demand in about 75% of scenarios.
- **Limitations:** only 28 scenarios per series; costs are user assumptions; perishability,
  batch sizes, existing stock and substitution between products are ignored.
"""
    )

# ------------------------------------------------------------ What-if
st.subheader("3 · What-if comparison")
custom = st.number_input("Add your own preparation quantity", min_value=0.0,
                         value=float(np.ceil(forecast)), step=1.0)
options = {
    "Forecast only": forecast,
    f"Forecast +{buffer}% buffer": prep,
    "Cost-balanced": opt,
    "Forecast +25%": forecast * 1.25,
    "Forecast −10%": forecast * 0.9,
    "Your quantity": custom,
}
rows = []
for name, q in options.items():
    r = core.simulate_outcomes(q, forecast, residuals, cu, co)
    rows.append({"Option": name, "Prepare": np.ceil(q),
                 "Avg units short": r["expected_shortage"], "Avg units wasted": r["expected_waste"],
                 "Stock-out chance": r["stockout_prob"] * 100, "Avg simulated cost": r["expected_cost"]})
table = pd.DataFrame(rows)
best = table["Avg simulated cost"].idxmin()

c1, c2 = st.columns([3, 2])
with c1:
    fig = go.Figure([
        go.Bar(name="Avg units short", x=table["Option"], y=table["Avg units short"],
               marker_color=ui.TAIL),
        go.Bar(name="Avg units wasted", x=table["Option"], y=table["Avg units wasted"],
               marker_color=ui.BASELINE),
    ])
    fig.update_layout(barmode="group", yaxis_title="Units (simulated average)",
                      title="Shortage vs waste for each option", hovermode="closest")
    st.plotly_chart(ui.style_fig(fig, 380), width="stretch")
with c2:
    qs = np.linspace(max(0, forecast * 0.5), forecast * 1.6 + 1, 120)
    costs = [core.simulate_outcomes(q, forecast, residuals, cu, co)["expected_cost"] for q in qs]
    f2 = go.Figure(go.Scatter(x=qs, y=costs, mode="lines", line=dict(color=ui.REFERENCE, width=3),
                              name="Avg simulated cost"))
    f2.add_vline(x=opt, line_dash="dash", line_color=ui.TAIL,
                 annotation_text="cost-balanced", annotation_position="top")
    f2.add_vline(x=forecast, line_dash="dot", line_color="gray",
                 annotation_text="forecast", annotation_position="bottom")
    f2.update_layout(title="Average simulated cost vs quantity", xaxis_title="Units prepared",
                     yaxis_title="Cost (assumed units)", showlegend=False, hovermode="x")
    st.plotly_chart(ui.style_fig(f2, 380), width="stretch")

st.dataframe(
    table.style.format({"Prepare": "{:,.0f}", "Avg units short": "{:,.1f}",
                        "Avg units wasted": "{:,.1f}", "Stock-out chance": "{:.0f}%",
                        "Avg simulated cost": "{:,.1f}"})
    .highlight_min(subset=["Avg simulated cost"], color="rgba(31,122,92,0.25)"),
    hide_index=True, width="stretch",
)
st.caption(
    f"Lowest simulated cost under your assumptions: **{table.loc[best, 'Option']}**. "
    "Change the shortage and waste costs to see how the balance shifts. These are model-based "
    "simulations, not measured savings."
)
