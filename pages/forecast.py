import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import foodflow_core as core
import ui

st.title("📈 Demand Forecasting")
st.write(
    "Pick a store, a product family, a forecast date and the promotion level. FOODFLOW builds "
    "the same lag, rolling and calendar features used in training and runs the frozen "
    "reference XGBoost model."
)

model, fcols = ui.model(), ui.feature_columns()
hist, source = ui.history()

# ------------------------------------------------------------ Inputs
with st.container(border=True):
    store, family = ui.series_picker("fc")
    c3, c4 = st.columns(2)
    first = core.LAST_OBSERVED_DATE + pd.Timedelta(days=1)
    last = core.LAST_OBSERVED_DATE + pd.Timedelta(days=core.MAX_HORIZON_DAYS)
    target = c3.date_input(
        "Forecast date", value=first.date(), min_value=first.date(), max_value=last.date(),
        help="Supported range: 2017-08-16 to 2017-08-31. The data ends on 2017-08-15, "
             "so 16 Aug is a true next-day forecast; later days are forecast recursively.",
    )
    promo = c4.number_input(
        "Items on promotion in this family", min_value=0, max_value=800, value=0, step=1,
        help="In this dataset `onpromotion` is the NUMBER of products in the family that are "
             "on promotion that day (median 0, 95th percentile 13, max 741).",
    )

horizon = (pd.Timestamp(target) - core.LAST_OBSERVED_DATE).days
series = core.series_history(hist, store, family)

try:
    fc = core.forecast_series(model, fcols, series, store, family, horizon, promo)
except ValueError as e:
    st.error(f"Forecast not generated: {e}")
    st.stop()

pred = fc["prediction"].iloc[-1]
direct = horizon == 1
val = ui.series_validation(store, family)
series_mae = core.mae(val["sales"], val["predicted"]) if len(val) else float("nan")
recent7 = series["sales"].tail(7).mean()
same_wd = series[series["date"] == pd.Timestamp(target) - pd.Timedelta(days=7 * ((horizon - 1) // 7 + 1))]

# ------------------------------------------------------------ Result
tag = "direct next-day forecast" if direct else f"recursive forecast, {horizon} days ahead"
st.markdown(
    f"""
<div class="ff-hero">
  <div>Predicted demand · <b>{family}</b> · Store {store} ·
  {pd.Timestamp(target).strftime('%A %d %B %Y')} <span class="ff-tag">{tag}</span></div>
  <div class="big">{pred:,.0f} units</div>
  <div>Typical error for this store-family in validation: ± {series_mae:,.0f} units (MAE over 28 days)</div>
</div>
""",
    unsafe_allow_html=True,
)

k = st.columns(4)
ui.kpi(k[0], "Last 7 days average", f"{recent7:,.0f}", "actual units / day")
ui.kpi(k[1], "Forecast vs 7-day average",
       f"{(pred / recent7 - 1) * 100:+.1f}%" if recent7 > 0 else "n/a")
if len(same_wd):
    ui.kpi(k[2], "Same weekday, last observed week", f"{same_wd['sales'].iloc[0]:,.0f}",
           same_wd["date"].iloc[0].strftime("%d %b %Y"))
else:
    ui.kpi(k[2], "Same weekday, last observed week", "n/a")
ui.kpi(k[3], "Inputs from real data", f"{fc['real_lag_share'].iloc[-1] * 100:.0f}%",
       "of the 7 lag / rolling features")

if not direct:
    st.warning(
        f"**Limitation:** the data ends on 2017-08-15, so for {pd.Timestamp(target):%d %b} the "
        "model's lag and rolling inputs partly use its own earlier predictions (recursive "
        "forecasting). The model was validated on next-day forecasts only, so treat this as "
        "less reliable than the 16 Aug forecast. The promotion value is applied to every "
        "intermediate day."
    )

# ------------------------------------------------------------ Chart
st.subheader("Recent demand and forecast")
fig = go.Figure()
fig.add_trace(go.Scatter(x=series["date"], y=series["sales"], name="Actual demand",
                         mode="lines+markers", line=dict(color=ui.ACTUAL, width=2),
                         marker=dict(size=4)))
if len(val):
    fig.add_trace(go.Scatter(x=val["date"], y=val["predicted"], name="Model on validation days",
                             mode="lines", line=dict(color=ui.REFERENCE, width=1.5, dash="dot")))
fig.add_trace(go.Scatter(x=[series["date"].iloc[-1], fc["date"].iloc[0]],
                         y=[series["sales"].iloc[-1], fc["prediction"].iloc[0]],
                         mode="lines", line=dict(color=ui.TAIL, width=1.5, dash="dot"),
                         showlegend=False, hoverinfo="skip"))
fig.add_trace(go.Scatter(x=fc["date"], y=fc["prediction"], name="Forecast (± validation MAE)",
                         mode="lines+markers", line=dict(color=ui.TAIL, width=2.5),
                         error_y=dict(type="constant", value=series_mae, thickness=1.2,
                                      color="rgba(217,130,43,0.6)")))
fig.add_trace(go.Scatter(x=[pd.Timestamp(target)], y=[pred], name="Selected date",
                         mode="markers", marker=dict(color=ui.TAIL, size=14, symbol="star")))
fig.add_vline(x=core.LAST_OBSERVED_DATE, line_dash="dash", line_color="gray")
fig.add_annotation(x=core.LAST_OBSERVED_DATE, y=1, yref="paper", text="last observed day",
                   showarrow=False, xanchor="right", font=dict(size=11, color="gray"))
fig.update_layout(yaxis_title="Units sold per day")
st.plotly_chart(ui.style_fig(fig, 420), width="stretch")
st.caption(
    f"History source: {source}. The dotted green line is what the same frozen model predicted "
    "for these days during validation (saved research output), so you can see how closely it "
    "tracks this particular series."
)

# ------------------------------------------------------------ Explanation
st.subheader("Why this number?")
X = core.feature_vector(series, store, family, promo, fcols).iloc[0]
e1, e2 = st.columns([3, 2])
with e1:
    weekday = pd.Timestamp(target).day_name()
    st.markdown(
        f"""
- The model's most important input is the **7-day rolling average** (about 60% of the
  model's total feature importance). For this series it was **{X['rolling_mean_7']:,.0f}**
  units/day going into 16 Aug.
- It also looks at the same weekday one and two weeks earlier (**lag 7 = {X['lag_7']:,.0f}**,
  **lag 14 = {X['lag_14']:,.0f}**), which captures the weekly shopping cycle — {weekday}s
  {'are weekend days, often busier' if pd.Timestamp(target).dayofweek >= 5 else 'are weekdays'}.
- Yesterday's sales (**lag 1 = {X['lag_1']:,.0f}**) and the 28-day average
  (**{X['rolling_mean_28']:,.0f}**) anchor the level.
- Calendar fields, the store number, the product family and the promotion count adjust the
  estimate further.
"""
    )
with e2:
    sens = core.promotion_sensitivity(model, fcols, series, store, family)
    fig2 = go.Figure(go.Bar(x=sens["onpromotion"].astype(str), y=sens["prediction"],
                            marker_color=ui.REFERENCE,
                            text=sens["prediction"].map(lambda x: f"{x:,.0f}"),
                            textposition="outside"))
    fig2.update_layout(title="16 Aug forecast vs promotion count",
                       xaxis_title="Items on promotion", yaxis_title="Predicted units",
                       hovermode="closest")
    st.plotly_chart(ui.style_fig(fig2, 300), width="stretch")
    st.caption("Everything else held fixed. Shows what the model learned, not a causal effect.")

with st.expander("Forecast path and exact model input"):
    t1, t2 = st.columns(2)
    show = fc.assign(date=fc["date"].dt.strftime("%a %d %b"),
                     real_lag_share=(fc["real_lag_share"] * 100).round(0).astype(int).astype(str) + "%")
    t1.dataframe(show.rename(columns={"real_lag_share": "real inputs"}), hide_index=True,
                 width="stretch",
                 column_config={"prediction": st.column_config.NumberColumn(format="%.1f")})
    t2.dataframe(X.rename("value").to_frame().loc[lambda d: ~d.index.str.startswith("family_")
                 | (d["value"] == 1)], width="stretch")
    t2.caption("Feature vector for the direct 16 Aug forecast (family one-hot columns that are 0 hidden).")
