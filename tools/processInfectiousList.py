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
    timestamps = [int(s) for s in infectiousList[1][0:-1].split(' ')]
    tsPos = binary_search(timestamps, timestamp)
    variants = [int(s) for s in infectiousList[2][0:-1].split(' ')]
    while variants[tsPos] != variant:
        if variants[tsPos] > variant:
            tsPos = tsPos-1
        else:
            tsPos = tsPos+1
    locationOffsets = [int(s) for s in infectiousList[3][0:-1].split(' ')]
    locationIds = [int(s) for s in infectiousList[4][0:-1].split(' ')]
    locationOffset = -1
    for i in range(locationOffsets[tsPos], locationOffsets[tsPos+1]):
        if locationIds[i] == infLocation:
            locationOffset = i
            break
    infectiousAgentOffsets = [int(s) for s in infectiousList[7][0:-1].split(' ')]
    infectiousAgentsAtLocation = [int(s) for s in infectiousList[8][0:-1].split(' ')]
    infectiousnessOfAgents = [float(s) for s in infectiousList[9][0:-1].split(' ')]
    outAgentIds = infectiousAgentsAtLocation[infectiousAgentOffsets[locationOffset]:infectiousAgentOffsets[locationOffset+1]]
    outInf = infectiousnessOfAgents[infectiousAgentOffsets[locationOffset]:infectiousAgentOffsets[locationOffset+1]]
    return outAgentIds, outInf

stats = open('stat.json')
people = json.load(stats)['Statistics']
stats.close()
dumpf = open('dumped.txt')
infectiousList = dumpf.read().splitlines()

#do the second person:
agentId = people[1]['ID']
infLocation = people[1]['InfectionLoc']
infTime = people[1]['infectionTime']
variant = people[1]['variant']

[IDList, InfList] = getContactList(agentId, infLocation, infTime, variant, infectiousList)
print(IDList)
print(InfList)