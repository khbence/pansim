#include "hip/hip_runtime.h"
#include "hip/hip_runtime.h"
#pragma once
#include "datatypes.h"
#include "timing.h"


#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_HIP
template<typename UnaryFunction, typename Count_t, typename PPState_t>
__global__ void reduce_by_location_kernel(unsigned* locationListOffsetsPtr,
    unsigned* locationAgentListPtr,
    Count_t* fullInfectedCountsPtr,
    PPState_t* PPValuesPtr,
    unsigned numLocations,
    UnaryFunction lam) {
    unsigned l = threadIdx.x + blockIdx.x * blockDim.x;
    if (l < numLocations) {
        for (unsigned agent = locationListOffsetsPtr[l]; agent < locationListOffsetsPtr[l + 1]; agent++) {
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
    if (agent < numAgents) { atomicAdd(&fullInfectedCountsPtr[agentLocationsPtr[agent]], lam(PPValuesPtr[agent])); }
}

#endif

class Util {
public:
    static int needAgentsSortedByLocation;
    static thrust::device_vector<unsigned> reduced_keys;
    static thrust::device_vector<unsigned> reduced_values_unsigned;
    static thrust::device_vector<float> reduced_values_float;
    template <typename Count_t>
    static thrust::device_vector<Count_t>& getBuffer(Count_t valtype);
    static void updatePerLocationAgentLists(const thrust::device_vector<unsigned>& locationOfAgents,
        thrust::device_vector<unsigned>& locationIdsOfAgents,
        thrust::device_vector<unsigned>& locationAgentList,
        thrust::device_vector<unsigned>& locationListOffsets);


template<typename UnaryFunction, typename Count_t, typename PPState_t>
static void reduce_by_location(thrust::device_vector<unsigned>& locationListOffsets,
    thrust::device_vector<unsigned>& locationAgentList,
    thrust::device_vector<Count_t>& fullInfectedCounts,
    thrust::device_vector<PPState_t>& PPValues,
    const thrust::device_vector<unsigned>& agentLocations,
    thrust::device_vector<unsigned>& locationIdsOfAgents,
    UnaryFunction lam) {
    unsigned numLocations = locationListOffsets.size() - 1;
    unsigned* locationListOffsetsPtr = thrust::raw_pointer_cast(locationListOffsets.data());
    Count_t* fullInfectedCountsPtr = thrust::raw_pointer_cast(fullInfectedCounts.data());
    PPState_t* PPValuesPtr = thrust::raw_pointer_cast(PPValues.data());
    const unsigned* agentLocationsPtr = thrust::raw_pointer_cast(agentLocations.data());
    unsigned* locationAgentListPtr = thrust::raw_pointer_cast(locationAgentList.data());
    unsigned numAgents = PPValues.size();
    
    PROFILE_FUNCTION();

    if (numLocations == 1) {
        fullInfectedCounts[0] = thrust::reduce(thrust::make_transform_iterator(PPValues.begin(), lam),
            thrust::make_transform_iterator(PPValues.end(), lam),
            (Count_t)0.0f);
    } else {

#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_OMP
#pragma omp parallel for
        for (unsigned l = 0; l < numLocations; l++) {
            for (unsigned agent = locationListOffsetsPtr[l]; agent < locationListOffsetsPtr[l + 1]; agent++) {
                fullInfectedCountsPtr[l] += lam(PPValuesPtr[locationAgentListPtr[agent]]);
            }
        }
#elif THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_HIP
#ifdef ATOMICS
        reduce_by_location_kernel_atomics<<<(numAgents - 1) / 256 + 1, 256>>>(
            agentLocationsPtr, fullInfectedCountsPtr, PPValuesPtr, numAgents, lam);
        hipDeviceSynchronize();
#else
        thrust::device_vector<Count_t>  &reduced_values2 = Util::getBuffer((Count_t)0);//(numLocations);//*reduced_values);
        if (reduced_keys.size()==0)reduced_keys.resize(numLocations);
        thrust::fill(reduced_keys.begin(), reduced_keys.end(), (unsigned)0);
        if (reduced_values2.size()==0)reduced_values2.resize(numLocations);
        thrust::reduce_by_key(locationIdsOfAgents.begin(), locationIdsOfAgents.end(), //keys (locationIDs sorted)
                              thrust::make_transform_iterator(
                                    thrust::make_permutation_iterator(PPValues.begin(), locationAgentList.begin()), lam), //values lam(PPValues[locationAgentList[i]])
                              reduced_keys.begin(), reduced_values2.begin());
        thrust::fill(fullInfectedCounts.begin(), fullInfectedCounts.end(), (Count_t)0);
        thrust::copy(reduced_values2.begin(), reduced_values2.end(), thrust::make_permutation_iterator(fullInfectedCounts.begin(), reduced_keys.begin()));
#endif
#endif
    }
}
};