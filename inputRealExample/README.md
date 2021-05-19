# Info about the file formats
By default these JSON files are compressed to save some storage for the repo. Don't forget to uncompress before usage.

## locations.json
It contains more information than necessary, but we stored these during data generation in case we need it. I'll list here the important attributes, the others can be any dummy value.
* id: unique for every locations (this will be referenced in the agents schedule)
* type: is the reference to the locationTypes.json's id field.
* state: can be ON/OFF, can be changed in runtime with the closure file
* infectious: how easily can the infectiousness transmit

## agents.json
* age: age of the agent
* sex: sex of the agent (F or M are supported at the moment)
* preCond: id of the pre-condition defined in parameters file
* typeId: reference to the agentTypes's id
* locations: need to give the actual location IDs that are defined for the agent type's schedule
    * typeID: reference to agentTypes.json/schedule/id
    * locID: reference to location.json/id