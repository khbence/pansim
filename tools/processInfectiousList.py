#!python
import json
import sys

def binary_search(arr, x):
    low = 0
    high = len(arr) - 1
    mid = 0
    while low <= high:
        mid = (high + low) // 2
        if arr[mid] < x:
            low = mid + 1
        elif arr[mid] > x:
            high = mid - 1
        else:
            return mid
    return -1

def getContactList(agentId, infLocation, timestamp, variant, infectiousList):
    timestamps = infectiousList[1]
    tsPos = binary_search(timestamps, timestamp)
    variants = infectiousList[2]
    while variants[tsPos] != variant:
        if variants[tsPos] > variant:
            tsPos = tsPos-1
        else:
            tsPos = tsPos+1
    locationOffsets = infectiousList[3]
    locationIds = infectiousList[4]
    locationOffset = -1
    for i in range(locationOffsets[tsPos], locationOffsets[tsPos+1]):
        if locationIds[i] == infLocation:
            locationOffset = i
            break
    infectiousAgentOffsets = infectiousList[7]
    infectiousAgentsAtLocation = infectiousList[8]
    infectiousnessOfAgents = infectiousList[9]
    outAgentIds = infectiousAgentsAtLocation[infectiousAgentOffsets[locationOffset]:infectiousAgentOffsets[locationOffset+1]]
    outInf = infectiousnessOfAgents[infectiousAgentOffsets[locationOffset]:infectiousAgentOffsets[locationOffset+1]]
    return outAgentIds, outInf

stats = open('stat.json')
people = json.load(stats)['Statistics']
stats.close()
dumpf = open('dumped.txt')
infectiousList = dumpf.read().splitlines()
for i in range(1,9):
    infectiousList[i] = [int(s) for s in infectiousList[i][0:-1].split(' ')]
infectiousList[9] = [float(s) for s in infectiousList[9][0:-1].split(' ')]

#do the second person:
agentId = people[1]['ID']
infLocation = people[1]['InfectionLoc']
infTime = people[1]['infectionTime']
#TODO: variant is now flags!
variant = people[1]['variant']

[IDList, InfList] = getContactList(agentId, infLocation, infTime, variant, infectiousList)
print(IDList)
print(InfList)