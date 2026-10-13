# Info about the file formats
By default these JSON files are compressed to save some storage for the repo. Don't forget to uncompress before usage.

## locations.json
It contains more information than necessary, but we stored these during data generation in case we need it. I'll list here the important attributes, the others can be any dummy value.
* id: unique for every locations (this will be referenced in the agents schedule)
* type: is the reference to the locationTypes.json's id field.
* state: can be ON/OFF, can be changed in runtime with the closure file
* infectious: how easily can the infectiousness transmit

## locations1.json and agents2.json

Built by `scripts/build_szeged_locations1.py` from `locations0.json` and `agents1.json`, using OpenStreetMap places inside the Szeged municipality.

`locations1.json` keeps every `locations0.json` place and adds the venues that file did not already have within 80 m (120 m for parks):

* type 5, evening food and drink: 8 places become 420 (restaurant, cafe, pub, bar, biergarten, fast food, food court)
* type 9, place of worship: 22 places become 52
* type 10, park (`leisure=park`): 15 places become 167

`agents2.json` is `agents1.json` with those three schedule slots reassigned. Each agent gets the nearest distinct places of that type. Distance is the minimum of the distance to a real home and a real workplace. The commuter boxes at coordinate `[120000, 740000]` are not used as anchors; agents who only have that box are anchored on their school. Pubs and bars are limited to ages 18–99. Other schedule slots are unchanged.

On the October 2022 agent file this puts the nearest food venue a median of about 100 m from home or work (90th percentile about 380 m), the nearest park about 160 m, and the nearest place of worship about 360 m. Counts and assignment loads are in `scripts/szeged_locations1_summary.json`.

## agents.json
* age: age of the agent
* sex: sex of the agent (F or M are supported at the moment)
* preCond: id of the pre-condition defined in parameters file
* typeId: reference to the agentTypes's id
* locations: need to give the actual location IDs that are defined for the agent type's schedule
    * typeID: reference to agentTypes.json/schedule/id
    * locID: reference to location.json/id