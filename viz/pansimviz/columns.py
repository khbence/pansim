"""Column catalog for panSim daily console output.

Most fields are tab-separated. MUT, IMM and INFV are comma-separated
per-variant lists inside a single tab field. Length follows the number of
strains passed to --infectiousnessMultiplier (MUT omits the wild type, so it
is one shorter). Descriptions follow matlab/Readme/Compute_beta.txt, with the
simulator's actual accumulation where that note says "daily" but the printed
value is cumulative.
"""

from __future__ import annotations

from dataclasses import dataclass


EXPANDABLE = ("MUT", "IMM", "INFV")

# Stock compartments that partition the simulated population.
STOCK = ("S", "E", "I1", "I2", "I3", "I4", "I5_h", "I6_h", "R_h", "R", "D1", "D2")

VARIANT_NAMES = (
    "wild type",
    "variant 1",
    "variant 2",
    "variant 3",
    "variant 4",
    "variant 5",
    "variant 6",
)


@dataclass(frozen=True)
class Column:
    key: str
    label: str
    description: str
    group: str
    kind: str  # count | percent | ratio
    gt: str | None = None  # ground-truth series key, counts are population-scaled


COLUMNS: dict[str, Column] = {}


def _add(key, label, description, group, kind="count", gt=None):
    COLUMNS[key] = Column(key, label, description, group, kind, gt)


_add("S", "Susceptible", "People who can still catch the infection.", "Compartments")
_add("E", "Exposed (latent)", "Infected in the latent phase.", "Compartments")
_add("I1", "Presymptomatic", "Infected in the presymptomatic phase.", "Compartments")
_add("I2", "Asymptomatic", "Asymptomatic people in the main sequence of the disease.", "Compartments")
_add("I3", "Infectious, mild", "Infected people in the main sequence with mild symptoms.", "Compartments")
_add("I4", "Infectious, severe", "Infected people in the main sequence with severe symptoms.", "Compartments")
_add("I5_h", "Hospitalized, mild", "Hospitalized patients with mild symptoms.", "Hospital", gt="hospital_mild")
_add("I6_h", "Hospitalized, severe", "Hospitalized patients with severe symptoms.", "Hospital", gt="ventilated")
_add("R_h", "Recovered in hospital", "Recovered while recorded in hospital.", "Compartments")
_add("R", "Recovered", "Recovered.", "Compartments")
_add("D1", "COVID deaths", "Cumulative deaths due to COVID.", "Hospital", gt="deaths")
_add("D2", "Other deaths", "Cumulative deaths from other causes.", "Hospital")
_add("H", "In hospital (any recorded stay)", "People whose hospital stay is still open and who are not dead or susceptible.", "Hospital")
_add("T", "Random tests", "Daily tests excluding tests of symptomatic people at a doctor or hospital.", "Testing", gt="tests")
_add("P1", "Positives among random tests", "Daily positive tests out of T.", "Testing")
_add("P2", "Symptomatic positives", "Daily positive tests of symptomatic people at a doctor or hospital.", "Testing")
_add("Q", "Quarantined", "Number of people in quarantine.", "Control", gt="quarantine")
_add("QT", "Quarantined and infected", "People in quarantine who are infected.", "Control")
_add("NQ", "Infected, not quarantined", "Infected people who are not in quarantine.", "Control")
_add("HOM", "Stayed home", "People who stayed home all day (deaths removed).", "Control")
_add("VAC", "Vaccinations today", "Number of immunizations given today.", "Immunity", gt="vacc_new")
_add("NI", "New infections", "People whose infection timestamp falls on this day.", "Epidemic", gt="new_cases")
_add("INF", "Ever infected", "Cumulative number of agents with at least one infection.", "Epidemic", gt="registered")
_add("REINF", "Reinfection events", "Cumulative extra infections beyond the first (sum of infectedCount-1).", "Epidemic")
_add(
    "BSTR",
    "Booster doses",
    "Cumulative booster doses: sum over agents of immunizations beyond the first. The notes call this daily; the simulator prints the cumulative total.",
    "Immunity",
)
_add("HCI", "Healthcare workers affected (%)", "Percent of healthcare workers who are infected or quarantined.", "Hospital", "percent")
_add("HCE", "Healthcare workers newly infected", "Healthcare workers infected today.", "Hospital")
_add("INFH", "Infected in hospital locations", "Infected people whose current location type is hospital.", "Hospital")
_add("VNI", "Vaccinated, never infected", "Agents with at least one immunization and zero infections.", "Immunity")

for i in range(1, 7):
    _add(
        f"MUT{i}",
        f"Active infections, {VARIANT_NAMES[i]} (%)",
        "Share of currently infected people carrying this variant. The remainder is the wild type.",
        "Variants",
        "percent",
    )
    _add(
        f"IMM{i}",
        f"Immune to {VARIANT_NAMES[i - 1]}",
        "Sum of (1 - susceptibility) against this strain. Index 1 is the wild type.",
        "Immunity",
    )
    _add(
        f"INFV{i}",
        f"Ever infected with {VARIANT_NAMES[i - 1]}",
        "Cumulative agents whose infection history includes this strain. Index 1 is the wild type.",
        "Variants",
    )
_add(
    "IMM7",
    f"Immune to {VARIANT_NAMES[6]}",
    "Sum of (1 - susceptibility) against strain 7.",
    "Immunity",
)
_add(
    "INFV7",
    f"Ever infected with {VARIANT_NAMES[6]}",
    "Cumulative agents whose infection history includes strain 7.",
    "Variants",
)

# Derived series. Formulas are applied in metrics.py.
DERIVED: dict[str, Column] = {}


def _der(key, label, description, group, kind="count", gt=None):
    DERIVED[key] = Column(key, label, description, group, kind, gt)


_der("I_all", "All infected", "E + I1 + I2 + I3 + I4 + I5_h + I6_h.", "Epidemic", gt="active")
_der("I_sym", "Symptomatic infectious", "I3 + I4.", "Epidemic")
_der(
    "I_beta",
    "Infectious (beta weights)",
    "I1 + 0.75*I2 + I3 + I4 + 0.1*I5_h + 0.1*I6_h, the weights used in Compute_beta.txt.",
    "Epidemic",
)
_der("H_covid", "COVID hospital beds", "I5_h + I6_h.", "Hospital", gt="hospital")
_der("D_new", "New COVID deaths", "Day-to-day increase of D1.", "Hospital", gt="deaths_new")
_der("R_all", "Recovered (all)", "R + R_h.", "Compartments", gt="recovered")
_der("R_new", "New recoveries", "Day-to-day increase of R + R_h.", "Compartments", gt="recovered_new")
_der("P_all", "Positive tests", "P1 + P2.", "Testing", gt="positives")
_der("T_all", "All tests", "T + P2. P2 tests are the symptomatic tests excluded from T.", "Testing")
_der("VAC_cum", "People vaccinated (cumulative doses)", "Cumulative sum of daily vaccinations.", "Immunity", gt="vacc_cum")
_der("MUT_WT", "Active infections, wild type (%)", "100 minus the sum of MUT shares.", "Variants", "percent")
_der(
    "beta",
    "Transmission rate beta",
    "(NI / I_beta) * (N / S), with I_beta from Compute_beta.txt. Not population-scaled.",
    "Epidemic",
    "ratio",
)
_der("pos_rate", "Positive rate among random tests", "P1 / T. Not population-scaled.", "Testing", "ratio", gt="pos_rate")


def describe(key: str) -> Column:
    if key in DERIVED:
        return DERIVED[key]
    if key in COLUMNS:
        return COLUMNS[key]
    return Column(key, key, key, "Other", "count")


def all_metric_keys() -> list[str]:
    preferred = [
        "NI",
        "I_all",
        "H_covid",
        "I5_h",
        "I6_h",
        "D1",
        "D_new",
        "beta",
        "INF",
        "REINF",
        "S",
        "E",
        "I1",
        "I2",
        "I3",
        "I4",
        "R_all",
        "R",
        "R_h",
        "R_new",
        "D2",
        "Q",
        "QT",
        "NQ",
        "T",
        "P1",
        "P2",
        "P_all",
        "pos_rate",
        "VAC",
        "VAC_cum",
        "BSTR",
        "VNI",
        "MUT_WT",
        "MUT1",
        "MUT2",
        "MUT3",
        "MUT4",
        "MUT5",
        "MUT6",
        "INFV1",
        "INFV2",
        "INFV3",
        "INFV4",
        "INFV5",
        "INFV6",
        "INFV7",
        "IMM1",
        "IMM2",
        "IMM3",
        "IMM4",
        "IMM5",
        "IMM6",
        "IMM7",
        "HOM",
        "H",
        "HCI",
        "HCE",
        "INFH",
        "I_sym",
        "I_beta",
        "T_all",
    ]
    return preferred
