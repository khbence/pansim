# Info about the file formats
By default these JSOn files are compressed to save some storage for the repo. Don't forget to uncompress before usage.

## locations.json
* id: unique for every locations (this will be referenced in the agents schedule)
* type: is the reference to the locationTypes.json's id field.
* state: can be ON/OFF, I'm not sure if we'll need this, because I don't know how we'll change it's state during the simulation

## agents.json
* typeId: reference to the agentTypes's id
* locations/typeID: reference to agentTypes.json/schedule/id
* locations/locID: reference to location.json/id