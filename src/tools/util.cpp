#include "util.h"
#include "timing.h"
#include "thrust/set_operations.h"

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

void Util::updatePerLocationAgentListsSort(const thrust::device_vector<unsigned>& locationOfAgents,
    thrust::device_vector<unsigned>& futureCopyOfLocationOfAgents,
    thrust::device_vector<unsigned>& locationAgentList,
    thrust::device_vector<unsigned>& locationPartAgentList,
    thrust::device_vector<unsigned>& locationListOffsets) {
    //    PROFILE_FUNCTION();

//     thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(locationOfAgents.begin(), thrust::counting_iterator<unsigned>{0}))
//                     , thrust::make_zip_iterator(thrust::make_tuple(locationOfAgents.begin(), thrust::counting_iterator<unsigned>{static_cast<unsigned>(locationOfAgents.size())}))
//                     , thrust::make_zip_iterator(thrust::make_tuple(locationAgentList.begin(), locationPartAgentList.begin()))
//                     , [] HD (const thrust::tuple<unsigned, unsigned>& e) {
//                         return thrust::make_pair(thrust::get<0>(e), thrust::get<1>(e));
//                     });

// #if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
//     // Count number of people at any given location
//     thrust::fill(locationListOffsets.begin(), locationListOffsets.end(), 0);
//     reduce_by_location(locationListOffsets,
//         locationAgentList,
//         locationListOffsets,
//         locationAgentList /* will be unused */,
//         locationOfAgents,
//         [] HD(const unsigned& a) -> unsigned {
//             return unsigned(1);
//         });// This locationOfAgents maybe should be locationIdsOfAgents??
//     thrust::exclusive_scan(locationListOffsets.begin(), locationListOffsets.end(), locationListOffsets.begin());
// #else
//     // Now extract offsets into locationAgentList where locations begin
//     unsigned* locationIdsOfAgentsPtr = thrust::raw_pointer_cast(locationAgentList.data());
//     unsigned* locationListOffsetsPtr = thrust::raw_pointer_cast(locationListOffsets.data());
//     extractOffsets(locationIdsOfAgentsPtr, locationListOffsetsPtr, futureCopyOfLocationOfAgents.size(), locationListOffsets.size() - 1);
// #endif
};

void Util::updatePerLocationAgentListsSet(thrust::device_vector<unsigned>& locationAgentList,
    thrust::device_vector<unsigned>& locationPartAgentList,
    thrust::device_vector<unsigned>& scanResult,
    thrust::device_vector<unsigned>& flags,
    thrust::device_vector<thrust::tuple<unsigned, unsigned, unsigned>>& movement,
    thrust::device_vector<thrust::tuple<unsigned, unsigned>>& copyOfPair) {

    // thrust::transform(movement.begin(), movement.end(), flags.begin(), [] HD(const thrust::tuple<unsigned, unsigned, unsigned>& e) {
    //     return static_cast<unsigned>(thrust::get<1>(e) != thrust::get<2>(e));
    // });
    // thrust::exclusive_scan(flags.begin(), flags.end(), scanResult.begin());
    // auto movedLength = scanResult.back() + static_cast<unsigned>(flags.back());
    // thrust::device_vector<thrust::tuple<unsigned, unsigned, unsigned>> movedAgents(movedLength);
    // thrust::scatter_if(movement.begin(), movement.end(), scanResult.begin(), flags.begin(), movedAgents.begin());
    // thrust::sort(movedAgents.begin(), movedAgents.end(), [] HD(const thrust::tuple<unsigned, unsigned, unsigned>& lhs, const thrust::tuple<unsigned, unsigned, unsigned>& rhs) {
    //     if(thrust::get<1>(lhs) == thrust::get<1>(rhs)) {
    //         return thrust::get<0>(lhs) < thrust::get<0>(rhs);
    //     }
    //     return thrust::get<1>(lhs) < thrust::get<1>(rhs);
    // });
    // auto movementToPairFROM = [] HD(const thrust::tuple<unsigned, unsigned, unsigned>& e) {
    //     return thrust::make_tuple(thrust::get<1>(e), thrust::get<0>(e));
    // };
    // thrust::set_difference(thrust::make_zip_iterator(thrust::make_tuple(locationPartAgentList.begin(), locationAgentList.begin()))
    //     , thrust::make_zip_iterator(thrust::make_tuple(locationPartAgentList.end(), locationAgentList.end()))
    //     , thrust::make_transform_iterator(movedAgents.begin(), movementToPairFROM)
    //     , thrust::make_transform_iterator(movedAgents.end(), movementToPairFROM)
    //     , copyOfPair.begin());
    
    // thrust::sort(movedAgents.begin(), movedAgents.end(), [] HD(const thrust::tuple<unsigned, unsigned, unsigned>& lhs, const thrust::tuple<unsigned, unsigned, unsigned>& rhs) {
    //     if(thrust::get<2>(lhs) == thrust::get<2>(rhs)) {
    //         return thrust::get<0>(lhs) < thrust::get<0>(rhs);
    //     }
    //     return thrust::get<2>(lhs) < thrust::get<2>(rhs);
    // });
    // auto movementToPairTO = [] HD (const thrust::tuple<unsigned, unsigned, unsigned>& e) {
    //     return thrust::make_tuple(thrust::get<2>(e), thrust::get<0>(e));
    // };
    // thrust::set_union(copyOfPair.begin()
    //         , copyOfPair.begin() + static_cast<ptrdiff_t>(locationAgentList.size() - movedAgents.size())
    //         , thrust::make_transform_iterator(movedAgents.begin(), movementToPairTO)
    //         , thrust::make_transform_iterator(movedAgents.end(), movementToPairTO)
    //         , thrust::make_zip_iterator(thrust::make_tuple(locationPartAgentList.begin(), locationAgentList.begin())));
}