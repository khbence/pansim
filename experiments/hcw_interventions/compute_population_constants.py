"""One-off: compute the exact total population and healthcare-worker headcount for a given
agents/locations pair, replicating flagHealthcareWorkers() exactly
(src/runtime/simulationOrchestration.cpp:360-388): an agent is a healthcare worker if ANY of its
location-list entries has typeID==4 ("work", confirmed identical to the JSON's own typeID field -
see src/runtime/agentsListRuntime.cpp:138, `ts.push_back(l.typeID)`) AND that specific location's
true type (looked up by locID in locations.json, NOT the agent-declared typeID) is 12 (hospital)
or 14 (doctor/health centre).

These are simulation constants (fixed by the input population, not printed per-day in the tsv
output), needed to properly compare healthcare workers against the REST of the population
(excluding HCWs, not the whole population which already includes them) - see hcw/metrics.py.

IMPORTANT: --disableTourists defaults to 1 (include/agentsList.h:97-99) and none of this
experiment's scenarios override it, so agent typeID 9 ("tourist", src/runtime/agentsListRuntime.cpp:83)
never actually enters the simulation - it must be excluded here too, or this script's total won't
match the simulator's actual day-0 population (confirmed: 184745 in agents1.json, 5999 of them
typeID 9, leaving exactly 178746 - matching all_runs.tsv's actual day-0 S+E+...+D2 sum).

Usage: python compute_population_constants.py <agents.json> <locations.json>
For the validated dataset used by every scenario in this experiment:
    python compute_population_constants.py ../../inputRealExample/agents1.json ../../inputRealExample/locations0.json
"""
import json
import sys

agents_path, locations_path = sys.argv[1], sys.argv[2]
with open(locations_path) as f:
    loc_type = {p["ID"]: p["type"] for p in json.load(f)["places"]}
with open(agents_path) as f:
    people = [p for p in json.load(f)["people"] if p["typeID"] != 9]  # exclude disabled tourists

missing = 0
hcw = 0
for p in people:
    is_hcw = False
    for loc in p["locations"]:
        if loc["typeID"] == 4:
            t = loc_type.get(loc["locID"])
            if t is None:
                missing += 1
            elif t in (12, 14):
                is_hcw = True
                break
    hcw += is_hcw

print(f"total_population = {len(people)}")
print(f"healthcare_worker_count = {hcw} ({100 * hcw / len(people):.3f}%)")
if missing:
    print(f"WARNING: {missing} location references not found in {locations_path}")
