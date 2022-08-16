#include "util.h"
#include "timing.h"

#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA && defined(ATOMICS)
int Util::needAgentsSortedByLocation = 0;
#else
int Util::needAgentsSortedByLocation = 1;
#endif

thrust::device_vector<unsigned> Util::reduced_keys;
thrust::device_vector<float> Util::reduced_values_float;
thrust::device_vector<unsigned> Util::reduced_values_unsigned;

template <>
thrust::device_vector<float>& Util::getBuffer(float valtype) {
    return reduced_values_float;
}

template <>
thrust::device_vector<unsigned>& Util::getBuffer(unsigned valtype) {
    return reduced_values_unsigned;
}

#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
__global__ void extractOffsets_kernel(unsigned* locOfAgents, unsigned* locationListOffsets, unsigned length, unsigned nLocs) {
    unsigned i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i == 0)
        locationListOffsets[0] = 0;
    else if (i < length) {
        if (locOfAgents[i - 1] != locOfAgents[i]) {
            for (unsigned j = locOfAgents[i - 1] + 1; j <= locOfAgents[i]; j++) { locationListOffsets[j] = i; }
        }
        if (i == length - 1) {
            for (unsigned j = locOfAgents[length - 1] + 1; j <= nLocs; j++) { locationListOffsets[j] = length; }
        }
    }
}
#endif
void extractOffsets(unsigned* locOfAgents, unsigned* locationListOffsets, unsigned length, unsigned nLocs) {
#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_OMP
    locationListOffsets[0] = 0;
#pragma omp parallel for
    for (unsigned i = 1; i < length; i++) {
        if (locOfAgents[i - 1] != locOfAgents[i]) {
            for (unsigned j = locOfAgents[i - 1] + 1; j <= locOfAgents[i]; j++) { locationListOffsets[j] = i; }
        }
    }
    for (unsigned j = locOfAgents[length - 1] + 1; j <= nLocs; j++) { locationListOffsets[j] = length; }
    locationListOffsets[nLocs] = length;
#elif THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
    extractOffsets_kernel<<<(length - 1) / 256 + 1, 256>>>(locOfAgents, locationListOffsets, length, nLocs);
    cudaDeviceSynchronize();
#endif
}
void Util::updatePerLocationAgentLists(const thrust::device_vector<unsigned>& locationOfAgents,
    thrust::device_vector<unsigned>& locationIdsOfAgents,
    thrust::device_vector<unsigned>& locationAgentList,
    thrust::device_vector<unsigned>& locationListOffsets) {
    PROFILE_FUNCTION();

    // Make a copy of locationOfAgents
    thrust::copy(locationOfAgents.begin(), locationOfAgents.end(), locationIdsOfAgents.begin());
    thrust::sequence(locationAgentList.begin(), locationAgentList.end());
    // Now sort by location, so locationAgentList contains agent IDs sorted by
    // location
    if (Util::needAgentsSortedByLocation) {
        BEGIN_PROFILING("sort")
        thrust::stable_sort_by_key(locationIdsOfAgents.begin(), locationIdsOfAgents.end(), locationAgentList.begin());
        END_PROFILING("sort")
    }
#if (THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA)
    #ifdef ATOMICS
    // Count number of people at any given location
    thrust::fill(locationListOffsets.begin(), locationListOffsets.end(), 0);
    Util::reduce_by_location(locationListOffsets,
        locationAgentList,
        locationListOffsets,
        locationAgentList /* will be unused */,
        locationOfAgents,
        locationIdsOfAgents,
        [] HD(const unsigned& a) -> unsigned {
            return unsigned(1);
        });
    thrust::exclusive_scan(locationListOffsets.begin(), locationListOffsets.end(), locationListOffsets.begin());
    #else
    thrust::sequence(locationListOffsets.begin(), locationListOffsets.end());
    thrust::lower_bound(locationIdsOfAgents.begin(), locationIdsOfAgents.end(), locationListOffsets.begin(), locationListOffsets.end(), locationListOffsets.begin());
    #endif
#else
    // Now extract offsets into locationAgentList where locations begin
    unsigned* locationIdsOfAgentsPtr = thrust::raw_pointer_cast(locationIdsOfAgents.data());
    unsigned* locationListOffsetsPtr = thrust::raw_pointer_cast(locationListOffsets.data());
    extractOffsets(locationIdsOfAgentsPtr, locationListOffsetsPtr, locationIdsOfAgents.size(), locationListOffsets.size() - 1);
#endif
};