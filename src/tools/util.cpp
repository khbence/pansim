#include "util.h"
#include "timing.h"

#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
__global__ void extractOffsets_kernel(unsigned* locOfAgents,
    unsigned* locationListOffsets,
    unsigned length,
    unsigned nLocs) {
    unsigned i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i == 0)
        locationListOffsets[0] = 0;
    else if (i < length) {
        if (locOfAgents[i - 1] != locOfAgents[i]) {
            for (unsigned j = locOfAgents[i - 1] + 1; j <= locOfAgents[i]; j++) {
                locationListOffsets[j] = i;
            }
        }
        if (i == length - 1) {
            for (unsigned j = locOfAgents[length - 1] + 1; j <= nLocs; j++) {
                locationListOffsets[j] = length;
            }
        }
    }
}
#endif
void extractOffsets(unsigned* locOfAgents,
    unsigned* locationListOffsets,
    unsigned length,
    unsigned nLocs) {
#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_OMP
    locationListOffsets[0] = 0;
#pragma omp parallel for
    for (unsigned i = 1; i < length; i++) {
        if (locOfAgents[i - 1] != locOfAgents[i]) {
            for (unsigned j = locOfAgents[i - 1] + 1; j <= locOfAgents[i]; j++) {
                locationListOffsets[j] = i;
            }
        }
    }
    for (unsigned j = locOfAgents[length - 1] + 1; j <= nLocs; j++) {
        locationListOffsets[j] = length;
    }
    locationListOffsets[nLocs] = length;
#elif THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
    extractOffsets_kernel<<<(length - 1) / 256 + 1, 256>>>(
        locOfAgents, locationListOffsets, length, nLocs);
    cudaDeviceSynchronize();
#endif
}
void Util::updatePerLocationAgentLists(const thrust::device_vector<unsigned>& locationOfAgents,
    thrust::device_vector<unsigned>& locationIdsOfAgents,
    thrust::device_vector<unsigned>& locationAgentList,
    thrust::device_vector<unsigned>& locationListOffsets) {
//    PROFILE_FUNCTION();

    // Make a copy of locationOfAgents
    thrust::copy(locationOfAgents.begin(), locationOfAgents.end(), locationIdsOfAgents.begin());
    thrust::sequence(locationAgentList.begin(), locationAgentList.end());
    // Now sort by location, so locationAgentList contains agent IDs sorted by
    // location
    //BEGIN_PROFILING("sort")
    thrust::stable_sort_by_key(
        locationIdsOfAgents.begin(), locationIdsOfAgents.end(), locationAgentList.begin());
    //END_PROFILING("sort")
#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
    //Count number of people at any given location
    thrust::fill(locationListOffsets.begin(), locationListOffsets.end(), 0);
    reduce_by_location(locationListOffsets,
                       locationAgentList,
                       locationListOffsets, locationAgentList /* will be unused */,
                       locationOfAgents, [] HD (const unsigned &a) -> unsigned {return unsigned(1);}); //This locationOfAgents maybe should be locationIdsOfAgents??
    thrust::exclusive_scan(locationListOffsets.begin(), locationListOffsets.end(), locationListOffsets.begin());
#else
    // Now extract offsets into locationAgentList where locations begin
    unsigned* locationIdsOfAgentsPtr = thrust::raw_pointer_cast(locationIdsOfAgents.data());
    unsigned* locationListOffsetsPtr = thrust::raw_pointer_cast(locationListOffsets.data());
    extractOffsets(locationIdsOfAgentsPtr,
        locationListOffsetsPtr,
        locationIdsOfAgents.size(),
        locationListOffsets.size() - 1);
#endif
    
};