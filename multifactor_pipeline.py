#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Condensed Multi-Factor Strategy (Momentum + Low Volatility)
- Monthly rebalanced, equal-weight Top-N portfolios
- Costs from turnover (round-trip bps)
- Benchmarked to SPY
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# --------------------------- Config ---------------------------
TICKERS = [
    "AAPL","MSFT","NVDA","AMZN","GOOGL","META","AVGO","TSLA",
    "LLY","HD","KO","MCD","COST","WMT","PEP","BAC","JPM","V","MA","XOM"
]
START = "2019-01-01"          # backtest start
END   = None                  # to latest
TOP_N = 5                     # number of names per portfolio
COST_BPS_ROUNDTRIP = 20.0     # round-trip trading cost in bps per monthly rebalance
RF_ANNUAL = 0.02              # risk-free assumption for Sharpe
MOM_LB_MONTHS = 12            # 12-1 momentum lookback (skip handled via shift)
LV_LB_DAYS = 126              # ~6m of trading days for vol
COMBO_WEIGHT_MOM = 0.5        # 50/50 blend (1.0 = pure Momentum, 0.0 = pure Low-Vol)

# ------------------------ Small utilities ---------------------
def resample_monthly_last(px: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Business-day fill forward, monthly end (ME) last price."""
    return px.asfreq("B").ffill().resample("ME").last()

def to_series(x) -> pd.Series:
    """Robustly coerce 1-D data to Series."""
    if x is None:
        return None
    if isinstance(x, pd.Series):
        return x.dropna()
    if isinstance(x, pd.DataFrame):
        if x.shape[1] == 1:
            return x.iloc[:, 0].dropna()
        raise ValueError("Expected 1-column DataFrame.")
    arr = np.asarray(x).squeeze()
    return pd.Series(arr).dropna()

# ------------------------ Factor signals ----------------------
def momentum_12_1(px_m: pd.DataFrame, lb: int = 12, skip: int = 1) -> pd.DataFrame:
    """12–1 momentum on monthly closes, shifted to skip the most recent month."""
    r12 = px_m.pct_change(lb)
    r1  = px_m.pct_change(1)
    mom = ((1 + r12) / (1 + r1) - 1).shift(skip)
    return mom

def low_vol_daily(px_d: pd.DataFrame, lb_days: int = 126) -> pd.DataFrame:
    """Negative trailing daily volatility (annualised) — higher is better."""
    vol = px_d.pct_change().rolling(lb_days).std() * np.sqrt(252)
    return -vol  # note the minus sign; higher = better

def xsec_rank(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional percentile ranks each month (higher = better)."""
    return df.replace([np.inf, -np.inf], np.nan).rank(axis=1, pct=True)

# -------------------- Portfolio construction ------------------
def _weights_from_holdings(holdings, universe) -> pd.Series:
    w = pd.Series(0.0, index=universe)
    if len(holdings):
        w.loc[holdings] = 1.0 / len(holdings)
    return w

def build_portfolio_with_costs(
    px_d: pd.DataFrame,
    score_m: pd.DataFrame,
    top_n: int = 5,
    tc_bps_roundtrip: float = 20.0
):
    """
    Monthly rebalance using prior month-end scores:
      - pick top_n by score, equal-weight for next month
      - one-way turnover = 0.5 * sum(|w_t - w_{t-1}|)
      - cost = turnover * (roundtrip_bps / 1e4)
    Returns: gross (Series), net (Series), holdings (list[dict]), turnover (Series)
    """
    mp = resample_monthly_last(px_d)
    mrets = mp.pct_change()
    months = score_m.index
    universe = mp.columns.tolist()

    gross_out, net_out, to_out, holds = [], [], [], []
    prev_w = pd.Series(0.0, index=universe)

    for i in range(1, len(months)):
        date, prev_date = months[i], months[i - 1]
        s = score_m.loc[prev_date].dropna()
        if len(s) < top_n:
            continue
        top = s.nlargest(top_n).index.tolist()
        new_w = _weights_from_holdings(top, universe)
        mret = mrets.loc[date, top].mean(skipna=True)

        turnover = 0.5 * (new_w - prev_w).abs().sum()
        cost = turnover * (tc_bps_roundtrip / 1e4)
        net = mret - cost

        gross_out.append((date, mret))
        net_out.append((date, net))
        to_out.append((date, turnover))
        holds.append({"date": date, "holdings": top})
        prev_w = new_w

    gross = pd.Series(dict(gross_out)).sort_index()
    net   = pd.Series(dict(net_out)).sort_index()
    to    = pd.Series(dict(to_out)).sort_index()
    return gross, net, holds, to

# --------------------- Performance analytics ------------------
def performance_stats(returns, bench=None, periods_per_year: int = 12, rf: float = 0.02) -> dict:
    r = to_series(returns)
    if r is None or r.empty:
        return {}
    total = (1 + r).prod() - 1
    ann   = (1 + total) ** (periods_per_year / len(r)) - 1
    vol   = r.std() * np.sqrt(periods_per_year)
    sharpe = (ann - rf) / vol if vol > 0 else 0.0
    cum = (1 + r).cumprod()
    maxdd = (cum / cum.cummax() - 1).min()

    out = {
        "Total Return": f"{total:.1%}",
        "Annualized Return": f"{ann:.1%}",
        "Volatility": f"{vol:.1%}",
        "Sharpe": f"{sharpe:.2f}",
        "Max Drawdown": f"{maxdd:.1%}",
    }

    if bench is not None:
        b = to_series(bench).reindex(r.index).dropna()
        rb = r.reindex(b.index).dropna()
        b  = b.reindex(rb.index)
        if len(rb) >= 2:
            cov_rb = np.cov(rb.values, b.values, ddof=1)[0, 1]
            var_b  = np.var(b.values, ddof=1)
            beta = cov_rb / var_b if var_b != 0 else np.nan
            ex   = rb - b
            te   = ex.std() * np.sqrt(periods_per_year)
            ir   = (ex.mean() * periods_per_year / te) if te > 1e-12 else 0.0
        else:
            beta, te, ir = np.nan, np.nan, 0.0
        out.update({
            "Beta vs SPY": f"{beta:.2f}",
            "Tracking Error": "n/a" if np.isnan(te) else f"{te:.1%}",
            "Information Ratio": f"{ir:.2f}",
        })
    return out

def rolling_alpha(strategy: pd.Series, bench: pd.Series, window: int = 12) -> pd.Series:
    aligned = pd.concat([to_series(strategy), to_series(bench)], axis=1).dropna()
    if aligned.empty:
        return pd.Series(dtype=float)
    s, b = aligned.iloc[:, 0], aligned.iloc[:, 1]
    roll_s = (1 + s).rolling(window).apply(np.prod, raw=True) - 1
    roll_b = (1 + b).rolling(window).apply(np.prod, raw=True) - 1
    return (roll_s - roll_b).dropna()

# ------------------------------ Main --------------------------
def main():
    print("Downloading prices...")
    px_d = yf.download(TICKERS, start=START, end=END, auto_adjust=True, progress=False)["Close"]
    if isinstance(px_d, pd.Series):  # ensure DataFrame
        px_d = px_d.to_frame()

    # Benchmark (monthly % returns, named Series)
    spy_px = yf.download("SPY", start=START, end=END, auto_adjust=True, progress=False)["Close"]
    spy_m = resample_monthly_last(spy_px).pct_change().dropna().squeeze()
    spy_m.name = "SPY"

    # Signals (monthly)
    px_m = resample_monthly_last(px_d)
    mom_scores = momentum_12_1(px_m, lb=MOM_LB_MONTHS, skip=1)
    lv_scores  = low_vol_daily(px_d, lb_days=LV_LB_DAYS).resample("ME").last()

    # Percentile ranks and blend
    mom_rank = xsec_rank(mom_scores)
    lv_rank  = xsec_rank(lv_scores)
    combo = COMBO_WEIGHT_MOM * mom_rank + (1 - COMBO_WEIGHT_MOM) * lv_rank

    print("Building portfolios...")
    mom_gross, mom_net, mom_hold, mom_to = build_portfolio_with_costs(px_d, mom_rank, TOP_N, COST_BPS_ROUNDTRIP)
    lv_gross,  lv_net,  lv_hold,  lv_to  = build_portfolio_with_costs(px_d, lv_rank,  TOP_N, COST_BPS_ROUNDTRIP)
    cb_gross,  cb_net,  cb_hold,  cb_to  = build_portfolio_with_costs(px_d, combo,    TOP_N, COST_BPS_ROUNDTRIP)

    # Align benchmark to each series for stats
    strategies = {
        "Momentum (Net)": mom_net,
        "Low-Vol (Net)": lv_net,
        "Combo 50/50 (Net)": cb_net,
        "SPY": spy_m
    }

    print("\nPerformance (monthly basis):")
    print("=" * 70)
    rows = []
    for name, ret in strategies.items():
        if name == "SPY":
            stats = performance_stats(ret)
        else:
            stats = performance_stats(ret, spy_m.reindex(ret.index))
        rows.append({"Strategy": name, **stats})

    perf_df = pd.DataFrame(rows).set_index("Strategy")
    print(perf_df)

    # ---- Plots ----
    # 1) Cumulative performance (Net vs SPY)
    plt.figure(figsize=(12, 6))
    for name, ret in {"Momentum (Net)": mom_net, "Low-Vol (Net)": lv_net,
                      "Combo 50/50 (Net)": cb_net, "SPY": spy_m}.items():
        if len(ret) > 0:
            (1 + ret).cumprod().plot(label=name, linewidth=2)
    plt.title("Cumulative Performance (Monthly, Net)")
    plt.xlabel("Date"); plt.ylabel("Growth of $1")
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    plt.show()

    # 2) Rolling 12-month alpha (Net vs SPY)
    plt.figure(figsize=(12, 4))
    for name, ret in {"Momentum (Net)": mom_net, "Low-Vol (Net)": lv_net,
                      "Combo 50/50 (Net)": cb_net}.items():
        if len(ret) > 12:
            rolling_alpha(ret, spy_m.reindex(ret.index)).plot(label=name, linewidth=2)
    plt.axhline(0, linestyle="--", alpha=0.6)
    plt.title("Rolling 12-Month Alpha vs SPY (Net)")
    plt.xlabel("Date"); plt.ylabel("Alpha")
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    plt.show()

    # Latest holdings + average turnover
    latest = {
        "Momentum (Latest)": mom_hold[-1]["holdings"] if mom_hold else [],
        "Low-Vol (Latest)": lv_hold[-1]["holdings"] if lv_hold else [],
        "Combo (Latest)": cb_hold[-1]["holdings"] if cb_hold else [],
    }
    avg_to = {
        "Momentum Avg Turnover": float(mom_to.mean()) if not mom_to.empty else np.nan,
        "Low-Vol Avg Turnover": float(lv_to.mean())  if not lv_to.empty else np.nan,
        "Combo Avg Turnover": float(cb_to.mean())    if not cb_to.empty else np.nan,
    }

    print("\nLatest holdings:")
    for k, v in latest.items():
        print(f"  {k}: {', '.join(v) if v else '(n/a)'}")
    print("\nAverage monthly turnover:")
    for k, v in avg_to.items():
        print(f"  {k}: {v:.1%}" if pd.notna(v) else f"  {k}: n/a")

    # Optional CSV exports
    ret_df = pd.concat({
        "MOM_Net": mom_net, "LV_Net": lv_net, "Combo_Net": cb_net, "SPY": spy_m
    }, axis=1)
    ret_df.to_csv("monthly_returns.csv")
    perf_df.to_csv("performance_summary.csv")
    print("\nSaved: monthly_returns.csv, performance_summary.csv")

if __name__ == "__main__":
    main()


# In[13]:


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

    # Benchmark SPY — ensure Series + name
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


# In[19]:


get_ipython().run_cell_magic('writefile', 'multifactor_pipeline.py', '#!/usr/bin/env python3\n# -*- coding: utf-8 -*-\n"""\nmultifactor_pipeline.py — one-file pipeline + dashboard\n\nModes:\n  build     -> download data, build strategies, save artifacts & charts\n  ff        -> FF5 + Momentum regression (HAC), save betas CSV/PNG\n  onepager  -> assemble a one-pager PDF from artifacts\n  all       -> build + ff + onepager\n  app       -> Streamlit dashboard (run via: streamlit run multifactor_pipeline.py -- --mode app)\n"""\n\nfrom __future__ import annotations\nimport argparse\nfrom pathlib import Path\nimport numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\n\n# ----------------------- Config -----------------------\nART = Path("artifacts"); ART.mkdir(exist_ok=True)\nTICKERS = [\n    "AAPL","MSFT","NVDA","AMZN","GOOGL","META","AVGO","TSLA",\n    "LLY","HD","KO","MCD","COST","WMT","PEP","BAC","JPM","V","MA","XOM"\n]\nSTART = "2018-01-01"\nBENCH = "SPY"\nTOP_QUANTILE = 0.20          # top 20% long-only bucket\nTC_BPS_ROUNDTRIP = 0.0       # monthly turnover cost in bps (set 20.0 for 0.20%)\nRISK_FREE_ANNUAL = 0.02\n\n# -------------------- Utilities -----------------------\ndef _safe_imports_for_build():\n    import yfinance as yf  # lazy import so \'ff\'/\'onepager\' don\'t require yf\n    return yf\n\ndef resample_me(px: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:\n    return px.asfreq("B").ffill().resample("ME").last()\n\ndef rank_pct(df: pd.DataFrame, high_is_good: bool = True) -> pd.DataFrame:\n    r = df.replace([np.inf, -np.inf], np.nan).rank(axis=1, pct=True)\n    return r if high_is_good else 1 - r\n\ndef stats_monthly(r: pd.Series, b: pd.Series | None = None, rf_annual: float = RISK_FREE_ANNUAL) -> dict:\n    r = pd.Series(r).dropna()\n    if r.empty: return {}\n    total = (1 + r).prod() - 1\n    ann   = (1 + total) ** (12 / len(r)) - 1\n    vol   = r.std() * np.sqrt(12)\n    sharpe = (ann - rf_annual) / vol if vol > 0 else np.nan\n    out = {"Total Return": f"{total:.1%}", "Annualized Return": f"{ann:.1%}",\n           "Volatility": f"{vol:.1%}", "Sharpe Ratio": f"{sharpe:.2f}"}\n    if b is not None:\n        a = pd.concat([r, b], axis=1).dropna()\n        if len(a) > 2:\n            ex = a.iloc[:, 0] - a.iloc[:, 1]\n            te = ex.std() * np.sqrt(12)\n            ir = (ex.mean() * 12 / te) if te > 0 else np.nan\n            out |= {"Tracking Error": f"{te:.1%}", "Information Ratio": f"{ir:.2f}"}\n    return out\n\n# ---------------- Strategy building -------------------\ndef build_long_only_equal_weight(rets_m: pd.DataFrame, signal: pd.DataFrame, top_q: float = 0.20,\n                                 tc_bps_roundtrip: float = 0.0):\n    sig_lag = signal.shift(1)\n    top_mask = sig_lag.apply(lambda s: s >= s.quantile(1 - top_q), axis=1)\n    w = top_mask.div(top_mask.sum(axis=1), axis=0).fillna(0.0)\n    gross = (w * rets_m).sum(axis=1)\n    turnover = w.sub(w.shift(1)).abs().sum(axis=1) / 2.0\n    cost = turnover * (tc_bps_roundtrip / 1e4)\n    net = gross - cost\n    return net.rename("return"), turnover.rename("turnover")\n\ndef build_artifacts():\n    yf = _safe_imports_for_build()\n\n    # Universe prices\n    px = yf.download(TICKERS, start=START, auto_adjust=True, progress=False)["Close"]\n    if isinstance(px, pd.Series): px = px.to_frame()\n    px_m = resample_me(px)\n    rets_m = px_m.pct_change().dropna(how="all")\n\n    # Signals (monthly)\n    mom = ((1 + px_m.pct_change(12)) / (1 + px_m.pct_change(1)) - 1)\n    lv  = -(rets_m.rolling(6).std() * np.sqrt(12))\n\n    mom_sig = rank_pct(mom, True)\n    lv_sig  = rank_pct(lv,  True)\n\n    # Portfolios (long-only)\n    mom_net, mom_to = build_long_only_equal_weight(rets_m, mom_sig, TOP_QUANTILE, TC_BPS_ROUNDTRIP)\n    lv_net,  lv_to  = build_long_only_equal_weight(rets_m, lv_sig,  TOP_QUANTILE, TC_BPS_ROUNDTRIP)\n    combo_net = (0.5 * mom_net + 0.5 * lv_net).rename("COMBO_net")\n\n    # Benchmark SPY — ensure Series + name (fixes rename TypeError)\n    spy_px = yf.download(BENCH, start=START, auto_adjust=True, progress=False)["Close"]\n    if isinstance(spy_px, pd.DataFrame):\n        spy_px = spy_px.squeeze("columns")\n    spy = resample_me(spy_px).pct_change().dropna()\n    spy.name = "SPY"\n\n    # Align & save\n    aligned = pd.concat([mom_net.rename("MOM_net"),\n                         lv_net.rename("LV_net"),\n                         combo_net.rename("COMBO_net"),\n                         spy], axis=1).dropna()\n    aligned.to_parquet(ART / "aligned_returns.parquet")\n    aligned.to_csv(ART / "aligned_returns.csv")\n\n    # Performance table\n    rows = []\n    for col in ["MOM_net", "LV_net", "COMBO_net"]:\n        rows.append({"Strategy": col, **stats_monthly(aligned[col], aligned["SPY"])})\n    rows.append({"Strategy": "SPY", **stats_monthly(aligned["SPY"])})\n    perf = pd.DataFrame(rows).set_index("Strategy")\n    perf.to_parquet(ART / "perf_summary.parquet")\n    perf.to_csv(ART / "perf_summary.csv")\n\n    # Charts\n    cum = (1 + aligned[["COMBO_net", "SPY"]]).cumprod()\n    ax = cum.plot(figsize=(8, 3), linewidth=2); ax.grid(alpha=.3); ax.set_ylabel("Multiple")\n    ax.figure.tight_layout(); ax.figure.savefig(ART / "chart_cumulative.png", dpi=150); plt.close(ax.figure)\n\n    roll = (1 + aligned[["COMBO_net", "SPY"]]).rolling(12).apply(np.prod, raw=True) - 1\n    ex = (roll["COMBO_net"] - roll["SPY"]).dropna()\n    ax = ex.plot(figsize=(8, 3), linewidth=2); ax.axhline(0, ls="--", c="k", alpha=.5)\n    ax.grid(alpha=.3); ax.set_ylabel("Excess"); ax.figure.tight_layout()\n    ax.figure.savefig(ART / "chart_rolling_alpha.png", dpi=150); plt.close(ax.figure)\n\n    pd.concat({"MOM_turnover": mom_to, "LV_turnover": lv_to}, axis=1).to_csv(ART / "turnover.csv")\n    print("✅ Artifacts saved to ./artifacts")\n\n# ---------------- Factor attribution -----------------\ndef run_ff_regression():\n    import statsmodels.api as sm\n    from pandas_datareader import data as web\n\n    aligned = pd.read_parquet(ART / "aligned_returns.parquet")\n    aligned.index = pd.to_datetime(aligned.index)\n    best = aligned["COMBO_net"].rename("strat")\n\n    ff5 = web.DataReader("F-F_Research_Data_5_Factors_2x3", "famafrench")[0]\n    mom = web.DataReader("F-F_Momentum_Factor", "famafrench")[0]\n    ff  = ff5.join(mom, how="inner")\n    # convert % -> decimals once if needed\n    if (ff.abs().mean() > 1).any():\n        ff = ff / 100.0\n    ff.index = ff.index.to_timestamp("M")\n    ff.columns = (ff.columns.str.strip()\n                  .str.replace("Mkt-RF", "MktRF", regex=False)\n                  .str.replace("Mom.", "Mom", regex=False))\n\n    df = pd.concat([best, ff], axis=1, join="inner").dropna()\n    if df.empty: raise RuntimeError("No overlap between strategy and FF factors.")\n\n    y = df["strat"] - df["RF"]\n    X = sm.add_constant(df[["MktRF","SMB","HML","RMW","CMA","Mom"]])\n    ols = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 6})\n\n    betas = ols.params.rename({"const": "Alpha"}).to_frame("coef")\n    betas.index.name = "factor"\n    betas.to_csv(ART / "ff5_mom_betas.csv", index=True)\n\n    order = [c for c in ["Alpha","MktRF","SMB","HML","RMW","CMA","Mom"] if c in betas.index]\n    ax = betas.loc[order, "coef"].plot(kind="bar", figsize=(7, 3), title="FF5 + Momentum Coefficients")\n    ax.grid(alpha=.3); plt.tight_layout()\n    plt.savefig(ART / "ff5_mom_betas.png", dpi=160); plt.close()\n\n    alpha = betas.loc["Alpha", "coef"] if "Alpha" in betas.index else 0.0\n    alpha_annual = (1 + alpha) ** 12 - 1\n    print(f"✅ Regression saved. Alpha (annual): {alpha_annual:.2%} | R²: {ols.rsquared:.3f}")\n\n# ---------------- One-pager PDF ----------------------\ndef make_one_pager():\n    from matplotlib.backends.backend_pdf import PdfPages\n    perf_path, cum_png, roll_png, betas_png = ART/"perf_summary.csv", ART/"chart_cumulative.png", ART/"chart_rolling_alpha.png", ART/"ff5_mom_betas.png"\n    out_pdf = ART / "one_pager_report.pdf"\n    if not perf_path.exists():\n        raise FileNotFoundError("Missing artifacts/perf_summary.csv. Run --mode build first.")\n    perf = pd.read_csv(perf_path)\n\n    with PdfPages(out_pdf) as pdf:\n        fig = plt.figure(figsize=(11, 8.5))\n        plt.figtext(0.05, 0.95, "Multi-Factor Equity Strategy — One-Pager (Momentum & Low-Vol)", fontsize=16, fontweight="bold")\n        plt.figtext(0.05, 0.92, "Period: monthly; Benchmark: SPY | Metrics: CAGR/Sharpe/MaxDD/Beta/TE/IR", fontsize=10, color="dimgray")\n\n        axT = plt.axes([0.05, 0.52, 0.43, 0.36]); axT.axis("off")\n        cols_pref = ["Strategy","Total Return","Annualized Return","Volatility","Sharpe Ratio","Max Drawdown","Tracking Error","Information Ratio"]\n        cols = [c for c in cols_pref if c in perf.columns]\n        table = axT.table(cellText=perf[cols].values, colLabels=cols, loc="center", cellLoc="center")\n        table.auto_set_font_size(False); table.set_fontsize(9); table.scale(1, 1.2)\n        axT.set_title("Performance Summary", fontsize=12, pad=6)\n\n        ax1 = plt.axes([0.52, 0.60, 0.43, 0.30]); ax1.axis("off")\n        ax1.imshow(plt.imread(cum_png)) if cum_png.exists() else ax1.text(0.5,0.5,"Missing chart_cumulative.png", ha="center", va="center")\n\n        ax2 = plt.axes([0.52, 0.22, 0.43, 0.30]); ax2.axis("off")\n        ax2.imshow(plt.imread(roll_png)) if roll_png.exists() else ax2.text(0.5,0.5,"Missing chart_rolling_alpha.png", ha="center", va="center")\n\n        ax3 = plt.axes([0.05, 0.08, 0.43, 0.28]); ax3.axis("off")\n        ax3.imshow(plt.imread(betas_png)) if betas_png.exists() else ax3.text(0.5,0.5,"Missing ff5_mom_betas.png", ha="center", va="center")\n\n        plt.figtext(0.05, 0.02, "Educational backtest. Past performance is not indicative of future results.", fontsize=8, color="gray")\n        pdf.savefig(fig); plt.close(fig)\n\n    print(f"✅ One-pager saved: {out_pdf}")\n\n# ---------------- Streamlit app ----------------------\ndef run_app():\n    import streamlit as st\n    st.set_page_config(page_title="Multi-Factor Strategy", layout="wide")\n    st.title("📈 Multi-Factor Equity Strategy Dashboard")\n\n    aligned_path, perf_path, betas_png = ART/"aligned_returns.parquet", ART/"perf_summary.parquet", ART/"ff5_mom_betas.png"\n    if not aligned_path.exists():\n        st.error("Missing artifacts/aligned_returns.parquet. Run: python multifactor_pipeline.py --mode build")\n        return\n\n    aligned = pd.read_parquet(aligned_path)\n    aligned.index = pd.to_datetime(aligned.index)\n    aligned = aligned.sort_index()\n\n    perf = pd.read_parquet(perf_path) if perf_path.exists() else None\n\n    st.sidebar.header("Controls")\n    bm_col = "SPY" if "SPY" in aligned.columns else aligned.columns[-1]\n    strat_cols = [c for c in aligned.columns if c != bm_col]\n    strat = st.sidebar.selectbox("Strategy", strat_cols or aligned.columns.tolist(), index=0)\n    win = st.sidebar.slider("Rolling window (months)", 6, 24, 12, 1)\n\n    st.subheader("Performance Summary")\n    if perf is not None and not perf.empty:\n        st.dataframe(perf.round(3), use_container_width=True)\n    else:\n        st.json(stats_monthly(aligned[strat], aligned.get(bm_col)))\n\n    st.subheader("Cumulative Growth of $1")\n    cum = (1 + aligned[[strat, bm_col]].dropna()).cumprod()\n    fig1, ax1 = plt.subplots(figsize=(8, 4))\n    cum.plot(ax=ax1, linewidth=2); ax1.grid(alpha=.3); ax1.set_ylabel("Multiple")\n    st.pyplot(fig1)\n\n    st.subheader(f"Rolling {win}-Month Excess vs {bm_col}")\n    both = aligned[[strat, bm_col]].dropna()\n    if len(both) >= win:\n        roll_s = (1 + both[strat]).rolling(win).apply(np.prod, raw=True) - 1\n        roll_b = (1 + both[bm_col]).rolling(win).apply(np.prod, raw=True) - 1\n        ex = (roll_s - roll_b).dropna()\n        fig2, ax2 = plt.subplots(figsize=(8, 3))\n        ex.plot(ax=ax2, linewidth=2); ax2.axhline(0, ls="--", c="k", alpha=.5); ax2.grid(alpha=.3); ax2.set_ylabel("Excess")\n        st.pyplot(fig2)\n    else:\n        st.info(f"Need at least {win} months.")\n\n    st.subheader("FF5 + Momentum Betas")\n    if betas_png.exists():\n        st.image(str(betas_png), use_column_width=False)\n    else:\n        st.info("Run factor regression: python multifactor_pipeline.py --mode ff")\n\n    colA, colB, colC = st.columns(3)\n    if (ART / "perf_summary.csv").exists():\n        with colA:\n            st.download_button("⬇️ perf_summary.csv",\n                               data=open(ART / "perf_summary.csv", "rb").read(),\n                               file_name="perf_summary.csv", mime="text/csv")\n    if (ART / "aligned_returns.csv").exists():\n        with colB:\n            st.download_button("⬇️ aligned_returns.csv",\n                               data=open(ART / "aligned_returns.csv", "rb").read(),\n                               file_name="aligned_returns.csv", mime="text/csv")\n    if (ART / "one_pager_report.pdf").exists():\n        with colC:\n            st.download_button("⬇️ one_pager_report.pdf",\n                               data=open(ART / "one_pager_report.pdf", "rb").read(),\n                               file_name="one_pager_report.pdf", mime="application/pdf")\n    st.caption("Educational backtest; not investment advice. © You")\n\n# ---------------- CLI entrypoint (positional + flag) --\ndef main():\n    choices = ["build","ff","onepager","all","app"]\n    p = argparse.ArgumentParser(description="All-in-one Multi-Factor (Momentum + Low-Vol) pipeline")\n    p.add_argument("-m","--mode", choices=choices, help="what to run")\n    p.add_argument("mode_pos", nargs="?", choices=choices, help="same as --mode (positional)")\n    args = p.parse_args()\n    mode = args.mode or args.mode_pos or "all"\n\n    if mode in ("build","all"):\n        build_artifacts()\n    if mode in ("ff","all"):\n        run_ff_regression()\n    if mode in ("onepager","all"):\n        make_one_pager()\n    if mode == "app":\n        run_app()\n\nif __name__ == "__main__":\n    main()\n')


# In[ ]:


get_ipython().system('python multifactor_pipeline.py build')
get_ipython().system('python multifactor_pipeline.py --mode ff')
get_ipython().system('python multifactor_pipeline.py all')
get_ipython().system('streamlit run multifactor_pipeline.py -- --mode app')

