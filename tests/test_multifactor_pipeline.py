import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
mfp = pytest.importorskip("multifactor_pipeline")


_weights_from_holdings = mfp._weights_from_holdings
build_portfolio_with_costs = mfp.build_portfolio_with_costs
performance_stats = mfp.performance_stats
resample_monthly_last = mfp.resample_monthly_last
to_series = mfp.to_series


@pytest.fixture
def synthetic_prices():
    dates = pd.date_range("2020-01-01", "2020-04-30", freq="B")
    steps = np.arange(len(dates))
    data = {
        "A": 100 * (1 + 0.01) ** steps,
        "B": 90 * (1 + 0.015) ** steps,
        "C": 80 * (1 + 0.005) ** steps,
    }
    return pd.DataFrame(data, index=dates)


def test_resample_monthly_last_forward_fills_business_days(synthetic_prices):
    resampled = resample_monthly_last(synthetic_prices)
    # January 2020 had its last business day on the 31st, so values should match.
    jan_last = synthetic_prices.loc["2020-01-31"]
    pd.testing.assert_series_equal(resampled.loc["2020-01-31"], jan_last)


def test_to_series_handles_various_inputs():
    series = pd.Series([1, 2, 3])
    df_single = pd.DataFrame({"x": [1, 2, 3]})
    array_input = [1, 2, 3]

    pd.testing.assert_series_equal(to_series(series), series)
    pd.testing.assert_series_equal(to_series(df_single), series)
    pd.testing.assert_series_equal(to_series(array_input), series)

    df_multi = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    with pytest.raises(ValueError):
        to_series(df_multi)


def test_weights_from_holdings_equal_weight():
    universe = ["A", "B", "C"]
    weights = _weights_from_holdings(["A", "C"], universe)
    expected = pd.Series({"A": 0.5, "B": 0.0, "C": 0.5}, dtype=float)
    pd.testing.assert_series_equal(weights, expected)


def test_build_portfolio_with_costs_matches_manual_calculation(synthetic_prices):
    months = pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31", "2020-04-30"])
    score_m = pd.DataFrame(
        {
            "A": [0.1, 0.6, 0.2, 0.3],
            "B": [0.3, 0.2, 0.5, 0.4],
            "C": [0.2, 0.4, 0.3, 0.1],
        },
        index=months,
    )

    gross, net, holdings, turnover = build_portfolio_with_costs(
        synthetic_prices, score_m, top_n=2, tc_bps_roundtrip=25.0
    )

    mp = resample_monthly_last(synthetic_prices)
    mrets = mp.pct_change()

    expected_dates = []
    expected_gross = []
    expected_net = []
    expected_turnover = []
    expected_holdings = []
    prev_weights = pd.Series(0.0, index=mp.columns)

    for i in range(1, len(months)):
        date = months[i]
        prev_date = months[i - 1]
        scores = score_m.loc[prev_date].dropna()
        if len(scores) < 2:
            continue
        top = scores.nlargest(2).index
        new_weights = _weights_from_holdings(top, mp.columns.tolist())
        turn = 0.5 * (new_weights - prev_weights).abs().sum()
        gross_ret = mrets.loc[date, top].mean(skipna=True)
        cost = turn * (25.0 / 1e4)

        expected_dates.append(date)
        expected_holdings.append(tuple(sorted(top)))
        expected_turnover.append(turn)
        expected_gross.append(gross_ret)
        expected_net.append(gross_ret - cost)
        prev_weights = new_weights

    pd.testing.assert_series_equal(
        gross, pd.Series(expected_gross, index=expected_dates), check_names=False
    )
    pd.testing.assert_series_equal(
        net, pd.Series(expected_net, index=expected_dates), check_names=False
    )
    pd.testing.assert_series_equal(
        turnover, pd.Series(expected_turnover, index=expected_dates), check_names=False
    )
    assert len(holdings) == len(expected_holdings)
    for record, expected_tuple, expected_date in zip(holdings, expected_holdings, expected_dates):
        assert record["date"] == expected_date
        assert tuple(sorted(record["holdings"])) == expected_tuple


def test_build_portfolio_with_costs_skips_when_not_enough_names(synthetic_prices):
    months = pd.to_datetime(["2020-01-31", "2020-02-29"])
    score_m = pd.DataFrame({"A": [0.1, np.nan], "B": [np.nan, np.nan]}, index=months)

    gross, net, holdings, turnover = build_portfolio_with_costs(
        synthetic_prices, score_m, top_n=2, tc_bps_roundtrip=10.0
    )

    assert gross.empty
    assert net.empty
    assert turnover.empty
    assert holdings == []


def test_performance_stats_with_benchmark():
    idx = pd.date_range("2021-01-31", periods=6, freq="M")
    returns = pd.Series([0.01, -0.02, 0.015, 0.005, 0.012, -0.004], index=idx)
    bench = pd.Series([0.008, -0.012, 0.01, 0.003, 0.009, -0.002], index=idx, name="SPY")

    stats = performance_stats(returns, bench)

    total = (1 + returns).prod() - 1
    ann = (1 + total) ** (12 / len(returns)) - 1
    vol = returns.std() * np.sqrt(12)
    sharpe = (ann - 0.02) / vol
    cum = (1 + returns).cumprod()
    maxdd = (cum / cum.cummax() - 1).min()

    aligned = pd.concat([returns, bench], axis=1).dropna()
    rb = aligned.iloc[:, 0]
    b = aligned.iloc[:, 1]
    cov_rb = np.cov(rb.values, b.values, ddof=1)[0, 1]
    var_b = np.var(b.values, ddof=1)
    beta = cov_rb / var_b
    ex = rb - b
    te = ex.std() * np.sqrt(12)
    ir = ex.mean() * 12 / te

    assert stats["Total Return"] == f"{total:.1%}"
    assert stats["Annualized Return"] == f"{ann:.1%}"
    assert stats["Volatility"] == f"{vol:.1%}"
    assert stats["Sharpe"] == f"{sharpe:.2f}"
    assert stats["Max Drawdown"] == f"{maxdd:.1%}"
    assert stats["Beta vs SPY"] == f"{beta:.2f}"
    assert stats["Tracking Error"] == f"{te:.1%}"
    assert stats["Information Ratio"] == f"{ir:.2f}"

