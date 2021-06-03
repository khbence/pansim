# General info
If a value is not relevant then for strings use an empty string.
If -1 is given for an interval right border it will set to infinit.

## agentTypes
This file stores information about the general behaviour of the different agent types. The actual locations for the schedule events will be defined in the agents file, or generated randomly if that was chosen.
* name: human readable ID (won't be used in simulation)
* id: unique id
* schedulesUnique: all the schedules for the given well-being (WB) states and day type
* schedule:
    * id: unique inside the type, but not throughout the types
    * type: reference to locationTypes' id
    * chance: chance of this event between the overlapping events
    * start/end: given in 24h hour format
    * duration: if shorter than end-start, then the actual schedule will be randomly chosen 

## closureRules
Describes the different events that can change the general state of the simulation.
* name: human readable ID (won't be used in simulation)
* conditionType: afterDays (enable after *threshold* days, for *openAfter* days), newDeadFraction (enable once the fraction of daily deaths w.r.t. total population exceeds *threshold* for *closeAfter* subsequent days. Disable once it goes below for *closeAfter* subsequent days), hospitalizedFraction (as with *newDeadFraction*, just with the fraction of people hospitalized w.r.t. total population).
* threshold: parameter in condition for enabling rule
* openAfter: see above
* closeAfter: see above. Has to be -1 for *openAfter*
* parameter: rule-specific parameter
* locationTypesToClose: if the rule affects locations, the list of location types

## configRandom
Defines different distributions for random data generation, which is used whne the agents and the locations files are not given, instead the agent number (*-n*) and the location number (*-N*) are given. If the *-r* flag is set, then the stateDistribution part will be used to set the agents' state randomly with this distribution. in all cases where we list chances the sum of those should be 1 (or empty list). *The current file is calculated from the realistic data town of Szeged (Hungary).*
* irregularLocationChance: What is the chance for an agent schedule location type, that the actual associated location is not the given type. *For example someone works in school, not studies there.*
* generalChance: not used in simulation, just an interesting statistic
* detailsOfChances: gives the information for all locations, please list all locations even if the chance is 0 for that.
    * value: the type of location
    * chanceForType: chance for the type
    * chanceFromAllIrregular: how many of all irregularities comes from this type (again, not used in simulation)
    * switchedToWhat: to which location can it be switched.
* locationTypeDistribution: Distribution of the location chances. List or locations.
* preCondDistribution: Distribution of different pre-conditions for the agents
* stateDistribution: Distribution of the states for the different age intervals. *As mentioned above this will be also use if -r parameter used.*
* agentTypeDistribution: Distribution of the agent types between agents.

## locationTypes
Indices of location types as used in the locations input
* publicSpace: outdoor public spaces
* home: residences
* hospital:
* doctor: doctors' offices
* school:
* classroom:
* work: offices, factories and similar, whose primary function is work as opposed to directly servicing visiting customers/visitors
* nurseryhome: elderly home
* types: It's not used in simulation, it just describes the location types in human readable form. Type indices have to match the above and the ones in the locations input

## parameters
Defines how the different agent states modifies the symptons and the transmission rate of an agent. Symptons will modifies the chance of a bad state progression for the agent, therefore lower number if increase the agent's survivalibility. The transmission rate will modify what infectiousness number would an infected agent to be considered. The sex and age part is self-explanatory. The possible preConditions are defined here and the ID are being used in agents file. The condition name is just a human readable description.

## Disease progression data


## Side notes
We will not expect the locations and agentType IDs to run from 0 to n in a monoton way without any gap. We will map it during initialization.
We will only expect it for the schedules of the AgentTypes to make continously through typic and common schedules.
