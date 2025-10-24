#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
multifactor_pipeline.py — one-file pipeline + dashboard

Modes:
  build     -> download data, build strategies, save artifacts & charts
  ff        -> FF5 + Momentum regression (HAC), save betas CSV/PNG
  onepager  -> assemble a one-pager PDF from artifacts
  all       -> build + ff + onepager
  app       -> Streamlit dashboard (run via: streamlit run multifactor_pipeline.py -- --mode app)
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------- Config -----------------------
ART = Path("artifacts"); ART.mkdir(exist_ok=True)
TICKERS = [
    "AAPL","MSFT","NVDA","AMZN","GOOGL","META","AVGO","TSLA",
    "LLY","HD","KO","MCD","COST","WMT","PEP","BAC","JPM","V","MA","XOM"
]
START = "2018-01-01"
BENCH = "SPY"
TOP_QUANTILE = 0.20          # top 20% long-only bucket
TC_BPS_ROUNDTRIP = 0.0       # monthly turnover cost in bps (set 20.0 for 0.20%)
RISK_FREE_ANNUAL = 0.02

# -------------------- Utilities -----------------------
def _safe_imports_for_build():
    import yfinance as yf  # lazy import so 'ff'/'onepager' don't require yf
    return yf

def resample_me(px: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    return px.asfreq("B").ffill().resample("ME").last()

def rank_pct(df: pd.DataFrame, high_is_good: bool = True) -> pd.DataFrame:
    r = df.replace([np.inf, -np.inf], np.nan).rank(axis=1, pct=True)
    return r if high_is_good else 1 - r

def stats_monthly(r: pd.Series, b: pd.Series | None = None, rf_annual: float = RISK_FREE_ANNUAL) -> dict:
    r = pd.Series(r).dropna()
    if r.empty: return {}
    total = (1 + r).prod() - 1
    ann   = (1 + total) ** (12 / len(r)) - 1
    vol   = r.std() * np.sqrt(12)
    sharpe = (ann - rf_annual) / vol if vol > 0 else np.nan
    out = {"Total Return": f"{total:.1%}", "Annualized Return": f"{ann:.1%}",
           "Volatility": f"{vol:.1%}", "Sharpe Ratio": f"{sharpe:.2f}"}
    if b is not None:
        a = pd.concat([r, b], axis=1).dropna()
        if len(a) > 2:
            ex = a.iloc[:, 0] - a.iloc[:, 1]
            te = ex.std() * np.sqrt(12)
            ir = (ex.mean() * 12 / te) if te > 0 else np.nan
            out |= {"Tracking Error": f"{te:.1%}", "Information Ratio": f"{ir:.2f}"}
    return out

# ---------------- Strategy building -------------------
def build_long_only_equal_weight(rets_m: pd.DataFrame, signal: pd.DataFrame, top_q: float = 0.20,
                                 tc_bps_roundtrip: float = 0.0):
    sig_lag = signal.shift(1)
    top_mask = sig_lag.apply(lambda s: s >= s.quantile(1 - top_q), axis=1)
    w = top_mask.div(top_mask.sum(axis=1), axis=0).fillna(0.0)
    gross = (w * rets_m).sum(axis=1)
    turnover = w.sub(w.shift(1)).abs().sum(axis=1) / 2.0
    cost = turnover * (tc_bps_roundtrip / 1e4)
    net = gross - cost
    return net.rename("return"), turnover.rename("turnover")

def build_artifacts():
    yf = _safe_imports_for_build()

    # Universe prices
    px = yf.download(TICKERS, start=START, auto_adjust=True, progress=False)["Close"]
    if isinstance(px, pd.Series): px = px.to_frame()
    px_m = resample_me(px)
    rets_m = px_m.pct_change().dropna(how="all")

    # Signals (monthly)
    mom = ((1 + px_m.pct_change(12)) / (1 + px_m.pct_change(1)) - 1)
    lv  = -(rets_m.rolling(6).std() * np.sqrt(12))

    mom_sig = rank_pct(mom, True)
    lv_sig  = rank_pct(lv,  True)

    # Portfolios (long-only)
    mom_net, mom_to = build_long_only_equal_weight(rets_m, mom_sig, TOP_QUANTILE, TC_BPS_ROUNDTRIP)
    lv_net,  lv_to  = build_long_only_equal_weight(rets_m, lv_sig,  TOP_QUANTILE, TC_BPS_ROUNDTRIP)
    combo_net = (0.5 * mom_net + 0.5 * lv_net).rename("COMBO_net")

    # Benchmark SPY — ensure Series + name (fixes rename TypeError)
    spy_px = yf.download(BENCH, start=START, auto_adjust=True, progress=False)["Close"]
    if isinstance(spy_px, pd.DataFrame):
        spy_px = spy_px.squeeze("columns")
    spy = resample_me(spy_px).pct_change().dropna()
    spy.name = "SPY"

    # Align & save
    aligned = pd.concat([mom_net.rename("MOM_net"),
                         lv_net.rename("LV_net"),
                         combo_net.rename("COMBO_net"),
                         spy], axis=1).dropna()
    aligned.to_parquet(ART / "aligned_returns.parquet")
    aligned.to_csv(ART / "aligned_returns.csv")

    # Performance table
    rows = []
    for col in ["MOM_net", "LV_net", "COMBO_net"]:
        rows.append({"Strategy": col, **stats_monthly(aligned[col], aligned["SPY"])})
    rows.append({"Strategy": "SPY", **stats_monthly(aligned["SPY"])})
    perf = pd.DataFrame(rows).set_index("Strategy")
    perf.to_parquet(ART / "perf_summary.parquet")
    perf.to_csv(ART / "perf_summary.csv")

    # Charts
    cum = (1 + aligned[["COMBO_net", "SPY"]]).cumprod()
    ax = cum.plot(figsize=(8, 3), linewidth=2); ax.grid(alpha=.3); ax.set_ylabel("Multiple")
    ax.figure.tight_layout(); ax.figure.savefig(ART / "chart_cumulative.png", dpi=150); plt.close(ax.figure)

    roll = (1 + aligned[["COMBO_net", "SPY"]]).rolling(12).apply(np.prod, raw=True) - 1
    ex = (roll["COMBO_net"] - roll["SPY"]).dropna()
    ax = ex.plot(figsize=(8, 3), linewidth=2); ax.axhline(0, ls="--", c="k", alpha=.5)
    ax.grid(alpha=.3); ax.set_ylabel("Excess"); ax.figure.tight_layout()
    ax.figure.savefig(ART / "chart_rolling_alpha.png", dpi=150); plt.close(ax.figure)

    pd.concat({"MOM_turnover": mom_to, "LV_turnover": lv_to}, axis=1).to_csv(ART / "turnover.csv")
    print("✅ Artifacts saved to ./artifacts")

# ---------------- Factor attribution -----------------
def run_ff_regression():
    import statsmodels.api as sm
    from pandas_datareader import data as web

    aligned = pd.read_parquet(ART / "aligned_returns.parquet")
    aligned.index = pd.to_datetime(aligned.index)
    best = aligned["COMBO_net"].rename("strat")

    ff5 = web.DataReader("F-F_Research_Data_5_Factors_2x3", "famafrench")[0]
    mom = web.DataReader("F-F_Momentum_Factor", "famafrench")[0]
    ff  = ff5.join(mom, how="inner")
    # convert % -> decimals once if needed
    if (ff.abs().mean() > 1).any():
        ff = ff / 100.0
    ff.index = ff.index.to_timestamp("M")
    ff.columns = (ff.columns.str.strip()
                  .str.replace("Mkt-RF", "MktRF", regex=False)
                  .str.replace("Mom.", "Mom", regex=False))

    df = pd.concat([best, ff], axis=1, join="inner").dropna()
    if df.empty: raise RuntimeError("No overlap between strategy and FF factors.")

    y = df["strat"] - df["RF"]
    X = sm.add_constant(df[["MktRF","SMB","HML","RMW","CMA","Mom"]])
    ols = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 6})

    betas = ols.params.rename({"const": "Alpha"}).to_frame("coef")
    betas.index.name = "factor"
    betas.to_csv(ART / "ff5_mom_betas.csv", index=True)

    order = [c for c in ["Alpha","MktRF","SMB","HML","RMW","CMA","Mom"] if c in betas.index]
    ax = betas.loc[order, "coef"].plot(kind="bar", figsize=(7, 3), title="FF5 + Momentum Coefficients")
    ax.grid(alpha=.3); plt.tight_layout()
    plt.savefig(ART / "ff5_mom_betas.png", dpi=160); plt.close()

    alpha = betas.loc["Alpha", "coef"] if "Alpha" in betas.index else 0.0
    alpha_annual = (1 + alpha) ** 12 - 1
    print(f"✅ Regression saved. Alpha (annual): {alpha_annual:.2%} | R²: {ols.rsquared:.3f}")

# ---------------- One-pager PDF ----------------------
def make_one_pager():
    from matplotlib.backends.backend_pdf import PdfPages
    perf_path, cum_png, roll_png, betas_png = ART/"perf_summary.csv", ART/"chart_cumulative.png", ART/"chart_rolling_alpha.png", ART/"ff5_mom_betas.png"
    out_pdf = ART / "one_pager_report.pdf"
    if not perf_path.exists():
        raise FileNotFoundError("Missing artifacts/perf_summary.csv. Run --mode build first.")
    perf = pd.read_csv(perf_path)

    with PdfPages(out_pdf) as pdf:
        fig = plt.figure(figsize=(11, 8.5))
        plt.figtext(0.05, 0.95, "Multi-Factor Equity Strategy — One-Pager (Momentum & Low-Vol)", fontsize=16, fontweight="bold")
        plt.figtext(0.05, 0.92, "Period: monthly; Benchmark: SPY | Metrics: CAGR/Sharpe/MaxDD/Beta/TE/IR", fontsize=10, color="dimgray")

        axT = plt.axes([0.05, 0.52, 0.43, 0.36]); axT.axis("off")
        cols_pref = ["Strategy","Total Return","Annualized Return","Volatility","Sharpe Ratio","Max Drawdown","Tracking Error","Information Ratio"]
        cols = [c for c in cols_pref if c in perf.columns]
        table = axT.table(cellText=perf[cols].values, colLabels=cols, loc="center", cellLoc="center")
        table.auto_set_font_size(False); table.set_fontsize(9); table.scale(1, 1.2)
        axT.set_title("Performance Summary", fontsize=12, pad=6)

        ax1 = plt.axes([0.52, 0.60, 0.43, 0.30]); ax1.axis("off")
        ax1.imshow(plt.imread(cum_png)) if cum_png.exists() else ax1.text(0.5,0.5,"Missing chart_cumulative.png", ha="center", va="center")

        ax2 = plt.axes([0.52, 0.22, 0.43, 0.30]); ax2.axis("off")
        ax2.imshow(plt.imread(roll_png)) if roll_png.exists() else ax2.text(0.5,0.5,"Missing chart_rolling_alpha.png", ha="center", va="center")

        ax3 = plt.axes([0.05, 0.08, 0.43, 0.28]); ax3.axis("off")
        ax3.imshow(plt.imread(betas_png)) if betas_png.exists() else ax3.text(0.5,0.5,"Missing ff5_mom_betas.png", ha="center", va="center")

        plt.figtext(0.05, 0.02, "Educational backtest. Past performance is not indicative of future results.", fontsize=8, color="gray")
        pdf.savefig(fig); plt.close(fig)

    print(f"✅ One-pager saved: {out_pdf}")

# ---------------- Streamlit app ----------------------
def run_app():
    import streamlit as st
    st.set_page_config(page_title="Multi-Factor Strategy", layout="wide")
    st.title("📈 Multi-Factor Equity Strategy Dashboard")

    aligned_path, perf_path, betas_png = ART/"aligned_returns.parquet", ART/"perf_summary.parquet", ART/"ff5_mom_betas.png"
    if not aligned_path.exists():
        st.error("Missing artifacts/aligned_returns.parquet. Run: python multifactor_pipeline.py --mode build")
        return

    aligned = pd.read_parquet(aligned_path)
    aligned.index = pd.to_datetime(aligned.index)
    aligned = aligned.sort_index()

    perf = pd.read_parquet(perf_path) if perf_path.exists() else None

    st.sidebar.header("Controls")
    bm_col = "SPY" if "SPY" in aligned.columns else aligned.columns[-1]
    strat_cols = [c for c in aligned.columns if c != bm_col]
    strat = st.sidebar.selectbox("Strategy", strat_cols or aligned.columns.tolist(), index=0)
    win = st.sidebar.slider("Rolling window (months)", 6, 24, 12, 1)

    st.subheader("Performance Summary")
    if perf is not None and not perf.empty:
        st.dataframe(perf.round(3), use_container_width=True)
    else:
        st.json(stats_monthly(aligned[strat], aligned.get(bm_col)))

    st.subheader("Cumulative Growth of $1")
    cum = (1 + aligned[[strat, bm_col]].dropna()).cumprod()
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    cum.plot(ax=ax1, linewidth=2); ax1.grid(alpha=.3); ax1.set_ylabel("Multiple")
    st.pyplot(fig1)

    st.subheader(f"Rolling {win}-Month Excess vs {bm_col}")
    both = aligned[[strat, bm_col]].dropna()
    if len(both) >= win:
        roll_s = (1 + both[strat]).rolling(win).apply(np.prod, raw=True) - 1
        roll_b = (1 + both[bm_col]).rolling(win).apply(np.prod, raw=True) - 1
        ex = (roll_s - roll_b).dropna()
        fig2, ax2 = plt.subplots(figsize=(8, 3))
        ex.plot(ax=ax2, linewidth=2); ax2.axhline(0, ls="--", c="k", alpha=.5); ax2.grid(alpha=.3); ax2.set_ylabel("Excess")
        st.pyplot(fig2)
    else:
        st.info(f"Need at least {win} months.")

    st.subheader("FF5 + Momentum Betas")
    if betas_png.exists():
        st.image(str(betas_png), use_column_width=False)
    else:
        st.info("Run factor regression: python multifactor_pipeline.py --mode ff")

    colA, colB, colC = st.columns(3)
    if (ART / "perf_summary.csv").exists():
        with colA:
            st.download_button("⬇️ perf_summary.csv",
                               data=open(ART / "perf_summary.csv", "rb").read(),
                               file_name="perf_summary.csv", mime="text/csv")
    if (ART / "aligned_returns.csv").exists():
        with colB:
            st.download_button("⬇️ aligned_returns.csv",
                               data=open(ART / "aligned_returns.csv", "rb").read(),
                               file_name="aligned_returns.csv", mime="text/csv")
    if (ART / "one_pager_report.pdf").exists():
        with colC:
            st.download_button("⬇️ one_pager_report.pdf",
                               data=open(ART / "one_pager_report.pdf", "rb").read(),
                               file_name="one_pager_report.pdf", mime="application/pdf")
    st.caption("Educational backtest; not investment advice. © You")

# ---------------- CLI entrypoint (positional + flag) --
def main():
    choices = ["build","ff","onepager","all","app"]
    p = argparse.ArgumentParser(description="All-in-one Multi-Factor (Momentum + Low-Vol) pipeline")
    p.add_argument("-m","--mode", choices=choices, help="what to run")
    p.add_argument("mode_pos", nargs="?", choices=choices, help="same as --mode (positional)")
    args = p.parse_args()
    mode = args.mode or args.mode_pos or "all"

    if mode in ("build","all"):
        build_artifacts()
    if mode in ("ff","all"):
        run_ff_regression()
    if mode in ("onepager","all"):
        make_one_pager()
    if mode == "app":
        run_app()

if __name__ == "__main__":
    main()
