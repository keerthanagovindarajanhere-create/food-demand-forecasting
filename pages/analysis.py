import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import foodflow_core as core
import ui

st.title("🔍 Demand & Error Analysis")
st.write(
    "All charts use the saved research outputs of the frozen model on the validation period "
    "(2017-07-19 to 2017-08-15). Nothing is retrained here."
)

v = ui.validation()
m = ui.headline_metrics()
families = sorted(v["family"].unique())

tab1, tab2, tab3, tab4 = st.tabs([
    "Demand trends", "Actual vs predicted", "Error concentration", "Regimes & tail-weighting",
])

# ============================================================ 1. Demand trends
with tab1:
    sel = st.multiselect("Product families (empty = all)", families,
                         default=["BEVERAGES", "GROCERY I", "PRODUCE", "DAIRY", "BREAD/BAKERY"])
    sub = v[v["family"].isin(sel)] if sel else v
    daily = sub.groupby(["date", "family"] if sel else "date", as_index=False)["sales"].sum()
    if sel:
        fig = px.line(daily, x="date", y="sales", color="family", markers=True)
    else:
        fig = px.line(daily, x="date", y="sales", markers=True)
        fig.update_traces(line_color=ui.ACTUAL)
    fig.update_layout(title="Total daily demand across all stores", yaxis_title="Units",
                      xaxis_title=None)
    st.plotly_chart(ui.style_fig(fig, 400), width="stretch")
    st.caption("Clear weekly cycle: weekends peak in almost every family.")

    fam = (v.groupby("family")
           .agg(avg_daily_demand=("sales", "mean"), mae=("absolute_error", "mean"))
           .reset_index().sort_values("avg_daily_demand", ascending=False))
    fam["mae_pct_of_demand"] = fam["mae"] / fam["avg_daily_demand"].replace(0, np.nan) * 100
    c1, c2 = st.columns(2)
    f1 = px.bar(fam.head(15), x="avg_daily_demand", y="family", orientation="h",
                color_discrete_sequence=[ui.ACTUAL])
    f1.update_layout(title="Average demand per store-day (top 15 families)",
                     xaxis_title="Units", yaxis_title=None, hovermode="closest")
    f1.update_yaxes(autorange="reversed")
    c1.plotly_chart(ui.style_fig(f1, 450), width="stretch")
    f2 = px.bar(fam.head(15), x="mae", y="family", orientation="h",
                color_discrete_sequence=[ui.REFERENCE], hover_data={"mae_pct_of_demand": ":.1f"})
    f2.update_layout(title="Model MAE for the same families", xaxis_title="MAE (units)",
                     yaxis_title=None, hovermode="closest")
    f2.update_yaxes(autorange="reversed")
    c2.plotly_chart(ui.style_fig(f2, 450), width="stretch")
    st.caption("High-volume families carry the largest absolute errors; hover to see MAE as a % of demand.")

# ============================================================ 2. Actual vs predicted
with tab2:
    c1, c2 = st.columns(2)
    fam_sel = c1.selectbox("Product family", ["All families"] + families)
    store_sel = c2.selectbox("Store", ["All stores"] + sorted(v["store_nbr"].unique().tolist()))
    sub = v
    if fam_sel != "All families":
        sub = sub[sub["family"] == fam_sel]
    if store_sel != "All stores":
        sub = sub[sub["store_nbr"] == store_sel]

    k = st.columns(4)
    ui.kpi(k[0], "Observations", f"{len(sub):,}")
    ui.kpi(k[1], "MAE", f"{core.mae(sub['sales'], sub['predicted']):,.2f}")
    ui.kpi(k[2], "RMSE", f"{core.rmse(sub['sales'], sub['predicted']):,.2f}")
    bias = (sub["predicted"] - sub["sales"]).mean()
    ui.kpi(k[3], "Mean bias", f"{bias:+,.2f}", "positive = over-prediction")
    st.write("")

    daily = sub.groupby("date", as_index=False)[["sales", "predicted"]].sum()
    fig = go.Figure([
        go.Scatter(x=daily["date"], y=daily["sales"], name="Actual", mode="lines+markers",
                   line=dict(color=ui.ACTUAL, width=2)),
        go.Scatter(x=daily["date"], y=daily["predicted"], name="Predicted (reference XGBoost)",
                   mode="lines+markers", line=dict(color=ui.REFERENCE, width=2, dash="dot")),
    ])
    fig.update_layout(title="Daily total: actual vs predicted", yaxis_title="Units")
    st.plotly_chart(ui.style_fig(fig, 380), width="stretch")

    c3, c4 = st.columns(2)
    sample = sub.sample(min(len(sub), 6000), random_state=0)
    lim = max(sample["sales"].max(), sample["predicted"].max()) * 1.05 if len(sample) else 1
    f3 = px.scatter(sample, x="sales", y="predicted", opacity=0.35,
                    hover_data=["date", "store_nbr", "family"],
                    color_discrete_sequence=[ui.REFERENCE])
    f3.add_trace(go.Scatter(x=[0, lim], y=[0, lim], mode="lines", name="perfect forecast",
                            line=dict(color="gray", dash="dash")))
    f3.update_layout(title="Each point = one store-family-day", xaxis_title="Actual",
                     yaxis_title="Predicted", hovermode="closest")
    c3.plotly_chart(ui.style_fig(f3, 400), width="stretch")
    if len(sub) > len(sample):
        c3.caption(f"Random sample of {len(sample):,} of {len(sub):,} points for speed.")

    err = (sub["sales"] - sub["predicted"])
    clip = err.abs().quantile(0.99) if len(err) else 1
    f4 = px.histogram(err.clip(-clip, clip), nbins=80, color_discrete_sequence=[ui.TAIL])
    f4.update_layout(title="Prediction error distribution (actual − predicted)",
                     xaxis_title="Error (units), clipped at the 99th percentile",
                     yaxis_title="Count", showlegend=False, hovermode="closest")
    c4.plotly_chart(ui.style_fig(f4, 400), width="stretch")

    st.markdown("**Largest individual errors in this selection**")
    st.dataframe(
        sub.nlargest(10, "absolute_error")[["date", "store_nbr", "family", "sales", "predicted",
                                            "absolute_error"]]
        .assign(date=lambda d: d["date"].dt.strftime("%Y-%m-%d (%a)")),
        hide_index=True, width="stretch",
        column_config={c: st.column_config.NumberColumn(format="%.0f")
                       for c in ["sales", "predicted", "absolute_error"]},
    )

# ============================================================ 3. Error concentration
with tab3:
    curve = core.cumulative_squared_error_share(v["sales"], v["predicted"])
    marks = [1, 2, 5, 10, 20]
    pts = pd.DataFrame({"obs_pct": marks, "share": [core.share_at(curve, p) for p in marks]})
    plot_curve = curve.iloc[np.unique(np.r_[np.linspace(0, len(curve) - 1, 2000).astype(int)])]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=plot_curve["obs_pct"], y=plot_curve["error_share_pct"],
                             mode="lines", line=dict(color=ui.REFERENCE, width=3),
                             name="Cumulative share of squared error"))
    fig.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode="lines", name="If errors were equal",
                             line=dict(color="gray", dash="dash")))
    fig.add_trace(go.Scatter(x=pts["obs_pct"], y=pts["share"], mode="markers+text",
                             text=[f"top {p}% → {s:.2f}%" for p, s in zip(marks, pts["share"])],
                             textposition="bottom right", marker=dict(size=10, color=ui.TAIL),
                             name="Reference points"))
    zoom = st.toggle("Zoom into the worst 25% of observations", value=True)
    fig.update_xaxes(title="Observations sorted from largest to smallest squared error (%)",
                     range=[0, 25] if zoom else [0, 100])
    fig.update_yaxes(title="Share of total squared error (%)", range=[0, 102])
    fig.update_layout(title="Cumulative squared-error contribution", hovermode="closest")
    st.plotly_chart(ui.style_fig(fig, 460), width="stretch")

    c1, c2 = st.columns([2, 3])
    c1.dataframe(pts.rename(columns={"obs_pct": "Top % of observations",
                                     "share": "Share of squared error (%)"}).round(2),
                 hide_index=True, width="stretch")
    top = v.nlargest(int(np.ceil(len(v) * 0.01)), "squared_error")
    comp = top["family"].value_counts().head(6)
    c1.caption(
        f"The top 1% ({len(top)} rows) are dominated by: "
        + ", ".join(f"{f} ({n})" for f, n in comp.items()) + "."
    )
    dg = ui.csv(str(core.RESULTS / "xgb_error_by_demand.csv"))
    dg["group"] = ["Very low", "Low", "Medium", "High", "Very high"][: len(dg)]
    dg["label"] = dg["group"] + "<br>" + dg["demand_group"]
    f2 = go.Figure(go.Bar(x=dg["label"], y=dg["MAE"], marker_color=ui.REFERENCE,
                          text=dg["MAE"].map(lambda x: f"{x:.2f}"), textposition="outside"))
    f2.update_layout(title="MAE by demand magnitude (quintiles of actual sales)",
                     yaxis_title="MAE (units)", hovermode="closest")
    c2.plotly_chart(ui.style_fig(f2, 360), width="stretch")
    st.info(
        f"Squared-error metrics such as RMSE are driven by a small set of difficult, high-demand "
        f"rows: the top 1% of observations explain {m['top1']:.2f}% of squared error, and the "
        f"highest-demand quintile has roughly {dg['MAE'].iloc[-1] / dg['MAE'].iloc[0]:.0f}× the "
        "MAE of the lowest. Average accuracy alone hides this."
    )

# ============================================================ 4. Regimes & tail-weighting
with tab4:
    rv = ui.csv(str(core.TAIL_DIR / "reference_vs_tail.csv"))
    st.markdown(
        "**Experiment.** Same data, split, features, hyperparameters and seed as the reference "
        "model; only the training sample weights changed. Rows with sales above the training "
        "80th percentile (298 units) got weight 2.0; all others 1.0."
    )
    show = rv[rv["metric"].isin(["MAE", "RMSE"])].copy()
    show["label"] = show["regime"].str.capitalize() + " " + show["metric"]
    fig = go.Figure([
        go.Bar(name="Reference XGBoost", x=show["label"], y=show["reference"],
               marker_color=ui.REFERENCE, text=show["reference"].map(lambda x: f"{x:.2f}"),
               textposition="outside"),
        go.Bar(name="Tail-weighted XGBoost", x=show["label"], y=show["tail_weighted"],
               marker_color=ui.TAIL, text=show["tail_weighted"].map(lambda x: f"{x:.2f}"),
               textposition="outside"),
    ])
    fig.update_layout(barmode="group", yaxis_title="Error (units, lower is better)",
                      title="Reference vs tail-weighted model by demand regime",
                      hovermode="closest")
    st.plotly_chart(ui.style_fig(fig, 420), width="stretch")

    table = rv.copy()
    table["regime"] = table["regime"].str.capitalize()
    table["better?"] = np.where(
        table["metric"].isin(["MAE", "RMSE", "Underprediction_Rate"]) & (table["change"] < 0), "✅ yes",
        np.where(table["metric"] == "Bias",
                 np.where(table["tail_weighted"].abs() < table["reference"].abs(), "✅ yes", "❌ no"),
                 "❌ no"))
    st.dataframe(table, hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="%.4f")
                                for c in ["reference", "tail_weighted", "change"]})
    st.success(
        "Overall MAE −0.64, overall RMSE −0.56 and high-demand MAE −2.73 units: a small but "
        "consistent gain where the error is concentrated."
    )
    st.warning(
        "Trade-offs: normal-demand RMSE rose by 1.47, normal MAE was essentially unchanged "
        "(+0.01), and high-demand over-prediction bias grew from +33.0 to +41.7 units. "
        "Tail-weighting is therefore a targeted improvement, not a universally better model. "
        "The app's forecasts use the frozen reference model."
    )

    fi = ui.csv(str(core.RESULTS / "xgb_feature_importance.csv")).head(12)
    f5 = px.bar(fi, x="importance", y="feature", orientation="h",
                color_discrete_sequence=[ui.REFERENCE])
    f5.update_layout(title="Reference model: top 12 features (XGBoost importance)",
                     yaxis_title=None, hovermode="closest")
    f5.update_yaxes(autorange="reversed")
    st.plotly_chart(ui.style_fig(f5, 380), width="stretch")

with st.expander("Original research figures (reports/research_analysis/figures)"):
    figs = sorted(core.FIGURES.glob("*.png"))
    if not figs:
        st.caption("No figure files found.")
    cols = st.columns(2)
    for i, p in enumerate(figs):
        cols[i % 2].image(str(p), caption=p.stem.replace("_", " "), width="stretch")
