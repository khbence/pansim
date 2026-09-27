import numpy as np

from pansimviz.aggregate import rolling_mean, summarize
from pansimviz.app import build_figure, scale_factors, simulation_dates
from pansimviz.ground_truth import align_to_dates, load_ground_truth
from pansimviz.metrics import prepare
from pansimviz.parse import parse_text

ROOT = "/home/ireguly/pansim"

SAMPLE = """noise before the table
S\tE\tI1\tI2\tI3\tI4\tI5_h\tI6_h\tR_h\tR\tD1\tD2\tH\tT\tP1\tP2\tQ\tQT\tNQ\tMUT\tHOM\tVAC\tNI\tINF\tREINF\tBSTR\tIMM\tHCI\tHCE\tINFV\tINFH\tVNI
178000\t100\t20\t10\t8\t4\t2\t1\t0\t50\t1\t0\t3\t10\t2\t1\t5\t4\t30\t10,0,0,0,0,0\t1000\t0\t15\t40\t0\t0\t200,200,200,200,200,200,200\t1\t2\t30,0,0,0,0,0,0\t4\t0
177900\t140\t30\t12\t9\t5\t3\t1\t0\t60\t2\t0\t4\t12\t3\t1\t6\t5\t40\t20,5,0,0,0,0\t900\t8\t20\t60\t1\t0\t260,260,260,260,260,260,260\t2\t1\t50,1,0,0,0,0,0\t5\t0
Timing report should be ignored
"""


def test_parser_expands_comma_fields_and_skips_prose():
    run = parse_text(SAMPLE)
    assert run is not None
    assert run.n_days == 2
    assert run.data["MUT1"].tolist() == [10.0, 20.0]
    assert run.data["MUT2"].tolist() == [0.0, 5.0]
    assert run.data["IMM1"].tolist() == [200.0, 260.0]
    assert len([c for c in run.columns if c.startswith("IMM")]) == 7
    assert len([c for c in run.columns if c.startswith("INFV")]) == 7
    assert run.data["NI"].tolist() == [15.0, 20.0]
    assert run.data["S"][0] == 178000


def test_derived_beta_matches_compute_beta_weights():
    from pansimviz.parse import RunSet

    run = prepare(RunSet("a", [parse_text(SAMPLE)])).runs[0]
    day0 = run.data
    weighted = day0["I1"][0] + 0.75 * day0["I2"][0] + day0["I3"][0] + day0["I4"][0] + 0.1 * day0["I5_h"][0] + 0.1 * day0["I6_h"][0]
    expected = (day0["NI"][0] / weighted) * (run.population() / day0["S"][0])
    assert np.isclose(day0["beta"][0], expected)
    assert day0["D_new"][1] == 1
    assert day0["H_covid"][0] == 3
    assert day0["MUT_WT"][0] == 90
    assert day0["I_all"][0] == 100 + 20 + 10 + 8 + 4 + 2 + 1


def test_ensemble_mean_and_std():
    from pansimviz.parse import RunSet

    second = SAMPLE.replace("\t15\t40\t", "\t25\t40\t").replace("\t20\t60\t", "\t30\t60\t")
    group = prepare(RunSet("g", [parse_text(SAMPLE), parse_text(second)]))
    summary = summarize(group, "NI", smoother=1)
    assert summary.mean.tolist() == [20.0, 25.0]
    assert np.isclose(summary.std[0], np.std([15, 25], ddof=1))
    smoothed = summarize(group, "NI", smoother=2)
    # trailing 2-day mean of 15,20 is 17.5 and of 25,30 is 27.5; ensemble mean 22.5
    assert np.isclose(smoothed.mean[1], 22.5)


def test_rolling_mean_skips_gaps():
    values = np.array([1.0, np.nan, 3.0, 4.0])
    out = rolling_mean(values, 2)
    assert np.isclose(out[0], 1.0)
    assert np.isclose(out[1], 1.0)
    assert np.isclose(out[2], 3.0)
    assert np.isclose(out[3], 3.5)


def test_ground_truth_dates_gaps_and_scale():
    gt = load_ground_truth(f"{ROOT}/korona_hun.xlsx")
    assert gt["date"][0] == np.datetime64("2020-03-04")
    assert gt["date"][-1] == np.datetime64("2022-12-28")
    # Early deaths are missing in the workbook.
    assert np.isnan(gt["deaths"][0])
    dates = simulation_dates(3, np.datetime64("2020-09-23"))
    assert str(dates[0]) == "2020-09-23"
    # A date before the workbook, and one inside it.
    before = align_to_dates(gt, np.array(["2019-01-01"], dtype="datetime64[D]"), "new_cases", 1.0)
    assert np.isnan(before[0])
    hit = align_to_dates(gt, np.array(["2020-03-04"], dtype="datetime64[D]"), "new_cases", 179500 / 9_600_000)
    assert np.isclose(hit[0], 2 * 179500 / 9_600_000)
    sim_mul, gt_mul = scale_factors("count", "simulation", 179500, 9_600_000)
    assert sim_mul == 1
    assert np.isclose(gt_mul, 179500 / 9_600_000)
    assert scale_factors("percent", "national", 179500, 9_600_000) == (1.0, 1.0)


def test_figure_has_mean_band_and_ground_truth():
    from pansimviz.parse import RunSet

    group = prepare(RunSet("baseline", [parse_text(SAMPLE), parse_text(SAMPLE.replace("\t15\t", "\t25\t"))]))
    gt = load_ground_truth(f"{ROOT}/korona_hun.xlsx")
    fig = build_figure(
        [group],
        "NI",
        start=np.datetime64("2020-09-23"),
        smoother=1,
        show_std=True,
        show_runs=True,
        show_gt=True,
        scale_mode="simulation",
        sim_pop=179500,
        national_pop=9_600_000,
        ground_truth=gt,
        title="New infections",
    )
    names = [t.name or "" for t in fig.data]
    assert any("mean" in n for n in names)
    assert any("Hungary" in n for n in names)
    # The national series does not connect across missing workbook cells.
    hungary = next(t for t in fig.data if "Hungary" in t.name)
    assert hungary.connectgaps is False


def test_saved_console_file_parses():
    run = __import__("pansimviz.parse", fromlist=["parse_file"]).parse_file(f"{ROOT}/sim1.tsv")
    assert run is not None
    assert run.n_days > 100
    assert "MUT1" in run.data and "INFV7" in run.data
    assert 179000 < run.population() < 180000
