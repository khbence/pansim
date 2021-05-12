#pragma once
#include "datatypes.h"
#include "timing.h"

class Util {
public:
    static void updatePerLocationAgentLists(const thrust::device_vector<unsigned>& locationOfAgents,
        thrust::device_vector<unsigned>& locationIdsOfAgents,
        thrust::device_vector<unsigned>& locationAgentList,
        thrust::device_vector<unsigned>& locationListOffsets);
};

#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
template<typename UnaryFunction, typename Count_t, typename PPState_t>
__global__ void reduce_by_location_kernel(unsigned* locationListOffsetsPtr,
    unsigned *locationAgentListPtr,
    Count_t* fullInfectedCountsPtr,
    PPState_t* PPValuesPtr,
    unsigned numLocations,
    UnaryFunction lam) {
    unsigned l = threadIdx.x + blockIdx.x * blockDim.x;
    if (l < numLocations) {
        for (unsigned agent = locationListOffsetsPtr[l]; agent < locationListOffsetsPtr[l + 1];
             agent++) {
            fullInfectedCountsPtr[l] += lam(PPValuesPtr[locationAgentListPtr[agent]]);
        }
    }
}
template<typename UnaryFunction, typename Count_t, typename PPState_t>
__global__ void reduce_by_location_kernel_atomics(const unsigned* agentLocationsPtr,
    Count_t* fullInfectedCountsPtr,
    PPState_t* PPValuesPtr,
    unsigned numAgents,
    UnaryFunction lam) {
    unsigned agent = threadIdx.x + blockIdx.x * blockDim.x;
    if (agent < numAgents) {
        atomicAdd(&fullInfectedCountsPtr[agentLocationsPtr[agent]], lam(PPValuesPtr[agent]));
    }
}

#endif
template<typename UnaryFunction, typename Count_t, typename PPState_t>
void reduce_by_location(thrust::device_vector<unsigned>& locationListOffsets,
    thrust::device_vector<unsigned>& locationAgentList,
    thrust::device_vector<Count_t>& fullInfectedCounts,
    thrust::device_vector<PPState_t>& PPValues,
    const thrust::device_vector<unsigned>& agentLocations,
    UnaryFunction lam) {
    unsigned numLocations = locationListOffsets.size() - 1;
    unsigned* locationListOffsetsPtr = thrust::raw_pointer_cast(locationListOffsets.data());
    Count_t* fullInfectedCountsPtr = thrust::raw_pointer_cast(fullInfectedCounts.data());
    PPState_t* PPValuesPtr = thrust::raw_pointer_cast(PPValues.data());
    const unsigned* agentLocationsPtr = thrust::raw_pointer_cast(agentLocations.data());
    unsigned* locationAgentListPtr = thrust::raw_pointer_cast(locationAgentList.data());
    unsigned numAgents = PPValues.size();

    //PROFILE_FUNCTION();

    if (numLocations == 1) {
        fullInfectedCounts[0] =
            thrust::reduce(thrust::make_transform_iterator(PPValues.begin(), lam),
                thrust::make_transform_iterator(PPValues.end(), lam),
                (Count_t)0.0f);
    } else {

#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_OMP
#pragma omp parallel for
        for (unsigned l = 0; l < numLocations; l++) {
            for (unsigned agent = locationListOffsetsPtr[l]; agent < locationListOffsetsPtr[l + 1];
                 agent++) {
                fullInfectedCountsPtr[l] += lam(PPValuesPtr[locationAgentListPtr[agent]]);
            }
        }
#elif THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
//#define ATOMICS
#ifdef ATOMICS
        reduce_by_location_kernel_atomics<<<(numAgents - 1) / 256 + 1, 256>>>(
            agentLocationsPtr, fullInfectedCountsPtr, PPValuesPtr, numAgents, lam);
#else
//#error "util.cpp's locationListOffsets computation CUDA pathway relies on atomics version, as this one needs locationListOffsets to already exist"
        reduce_by_location_kernel<<<(numLocations - 1) / 256 + 1, 256>>>(
            locationListOffsetsPtr, locationAgentListPtr, fullInfectedCountsPtr, PPValuesPtr, numLocations, lam);
#endif
        cudaDeviceSynchronize();
#endif
    }
}