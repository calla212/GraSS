#include <curand.h>
#include <curand_kernel.h>
#include <vector>


#include "bcg.cuh"
#include "utils.hpp"
#include "rand_num_gen.cuh"

#ifndef __NEXTDOOR_HPP__
#define __NEXTDOOR_HPP__

template<class SampleType, typename App>
struct NextDoorData {

  std::vector<SampleType> samples;
  std::vector<VertexID_t> hFinalSamples;
  std::vector<VertexID_t> initialContents;
  std::vector<VertexID_t> initialTransitToSampleValues;
  std::vector<int> devices;

  //Per Device Data.
  std::vector<SampleType*> dOutputSamples;
  std::vector<VertexID_t*> dSamplesToTransitMapKeys;
  std::vector<VertexID_t*> dSamplesToTransitMapValues;
  std::vector<VertexID_t*> dTransitToSampleMapKeys;
  std::vector<VertexID_t*> dTransitToSampleMapValues;
  std::vector<EdgePos_t*> dSampleInsertionPositions;
  std::vector<EdgePos_t*> dNeighborhoodSizes;
  std::vector<curandState*> dCurandStates;
  std::vector<size_t> maxThreadsPerKernel;
  std::vector<VertexID_t*> dFinalSamples;
  std::vector<VertexID_t*> dInitialSamples;
  int INVALID_VERTEX;
  int maxBits;
  std::vector<GPUBCGPartition> gpuBCGPartitions;

  VertexID_t n_nodes;
};

#endif