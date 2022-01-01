#include <iostream>
#include <algorithm>
#include <string>
#include <stdio.h>
#include <vector>
#include <bitset>
#include <unordered_set>
#include <time.h>
#include <sys/time.h>
#include <unistd.h>
#include <algorithm>
#include <numeric>
#include <string.h>
#include <assert.h>
#include <tuple>
#include <queue>
#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/device/device_select.cuh>
#include <cub/cub.cuh>
#include <curand.h>
#include <curand_kernel.h>
#include <cuda.h>

#include "graph.hpp"

#ifndef __NEXTDOOR_CU__
#define __NEXTDOOR_CU__

typedef VertexID VertexID_t;

#include "utils.hpp"
#include "rand_num_gen.cuh"
#include "libNextDoor.hpp"

using namespace utils;
using namespace GPUUtils;

const size_t N_THREADS = 256;


const int ALL_NEIGHBORS = -1;

const bool useGridKernel = true;
const bool useSubWarpKernel = false;
const bool useThreadBlockKernel = true;
const bool combineTwoSampleStores = true;

enum TransitKernelTypes {
  GridKernel = 1,
  ThreadBlockKernel = 2,
  SubWarpKernel = 3,
  IdentityKernel = 4,
  NumKernelTypes = 4
};

__constant__ char bcgPartitionBuff[sizeof(BCGPartition)];

template<typename App>
__host__ __device__
EdgePos_t subWarpSizeAtStep(int step)
{
  if (step == -1)
    return 0;
  
  //SubWarpSize is set to next power of 2
  
  EdgePos_t x = App().stepSize(step);

  if (x && (!(x&(x-1)))) {
    return x;
  } 

  x--;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  if (sizeof(EdgePos_t) == sizeof(uint64_t)) {
    //x |= x >> 32;
  }
  x++;

  return x;
}

template<typename App>
__host__ __device__
EdgePos_t stepSizeAtStep(int step)
{
  if (step == -1)
    return 0;

  EdgePos_t n = App().initialSampleSize(nullptr);
  for (int i = 0; i <= step; i++) {
    n = n * App().stepSize(i);
  }

  return n;
}

template<typename App>
__host__ __device__ int numberOfTransits(int step) {
  return App().stepSize(step);
}

__host__ __device__ bool isValidSampledVertex(VertexID_t neighbor, VertexID_t InvalidVertex) 
{
  return neighbor != InvalidVertex && neighbor != -1;
}

enum TransitParallelMode {
  //Describes the execution mode of Transit Parallel.
  NextFuncExecution, //Execute the next function
  CollectiveNeighborhoodSize, //Compute size of collective neighborhood
  CollectiveNeighborhoodComputation, //Compute the collective neighborhood 
};

#define STORE_TRANSIT_INDEX false
template<class SamplingType, typename App, TransitParallelMode tpMode, int CollNeighStepSize>
__global__ void samplingKernel(const int step, const size_t threadsExecuted, const size_t currExecutionThreads,
                               const VertexID_t deviceFirstSample, const VertexID_t invalidVertex,
                               const VertexID_t* transitToSamplesKeys, const VertexID_t* transitToSamplesValues,
                               const size_t transitToSamplesSize, SamplingType* samples, const size_t NumSamples,
                               VertexID_t* samplesToTransitKeys, VertexID_t* samplesToTransitValues,
                               VertexID_t* finalSamples, const size_t finalSampleSize, EdgePos_t* sampleInsertionPositions,
                               EdgePos_t* sampleNeighborhoodSizes, EdgePos_t* sampleNeighborhoodPos, 
                               VertexID_t* collectiveNeighborhoodCSRRows, 
                               EdgePos_t* collectiveNeighborhoodCSRCols, curandState* randStates) {
  EdgePos_t threadId = threadIdx.x + blockDim.x * blockIdx.x;

  if (threadId >= currExecutionThreads)
    return;
  
  curandState* randState = &randStates[threadId];

  threadId += threadsExecuted;

  EdgePos_t transitIdx = threadId/App().stepSize(step);
  EdgePos_t transitNeighborIdx = threadId % App().stepSize(step);
  EdgePos_t numTransits = numberOfTransits<App>(step);

  VertexID_t sampleIdx = transitToSamplesValues[transitIdx];
  assert(sampleIdx < NumSamples);
  VertexID_t transit = transitToSamplesKeys[transitIdx];
  VertexID_t neighbor = invalidVertex;
  BCGPartition* bcg = (BCGPartition*)&bcgPartitionBuff[0];

  if (transit != invalidVertex) {
    
    BCGVertex bcgv(transit, bcg->graph, bcg->offset[transit]);
    auto numTransitEdges = bcgv.outd;
    
    if (numTransitEdges != 0 && (tpMode == NextFuncExecution || tpMode == CollectiveNeighborhoodComputation)) {
      //Execute next in this mode only

      if (tpMode == NextFuncExecution) {
        // neighbor = 1;
        neighbor = App().next(step, &transit, sampleIdx, &samples[(sampleIdx - deviceFirstSample)], numTransitEdges, transitNeighborIdx, randState, &bcgv);
        // printf("Thread_%d %d -> %d\n", threadId, transit, neighbor);
      } else {
        int insertionPos = utils::atomicAdd(&sampleInsertionPositions[sampleIdx- deviceFirstSample], numTransitEdges);
        collectiveNeighborhoodCSRRows[(sampleIdx - deviceFirstSample)*App().initialSampleSize() + 0] = insertionPos;
      }
    } else if (tpMode == CollectiveNeighborhoodSize) {
      //Compute size of collective neighborhood for each sample.
      ::atomicAdd(&sampleNeighborhoodSizes[(sampleIdx - deviceFirstSample)], numTransitEdges);
    }
  }

  __syncwarp();
  if (tpMode == NextFuncExecution) {
    // EdgePos_t insertionPos = 0;
    //TODO: templatize over hasExplicitTransits()
    if (step != App().steps() - 1) {
      //No need to store at last step
      if (App().hasExplicitTransits()) {
        VertexID_t newTransit = App().stepTransits(step+1, sampleIdx, samples[(sampleIdx - deviceFirstSample)], threadId%numTransits, randState);
        samplesToTransitValues[threadId] = newTransit != -1 ? newTransit : invalidVertex;
      } else {
        samplesToTransitValues[threadId] = neighbor != -1 ? neighbor : invalidVertex;
      }
      samplesToTransitKeys[threadId] = sampleIdx;
    }

    // if (numberOfTransits<App>(step) > 1 && isValidSampledVertex(neighbor, invalidVertex)) {   
    //   //insertionPos = finalSampleSizeTillPreviousStep + transitNeighborIdx; //
    //   if (step == 0) {
    //     insertionPos = transitNeighborIdx;
    //   } else {
    //     size_t finalSampleSizeTillPreviousStep = 0;
    //     size_t neighborsToSampleAtStep = 1;
    //     for (int _s = 0; _s < step; _s++) {
    //       neighborsToSampleAtStep *= App().stepSize(_s);
    //       finalSampleSizeTillPreviousStep += neighborsToSampleAtStep;
    //     }
    //     insertionPos = finalSampleSizeTillPreviousStep + utils::atomicAdd(&sampleInsertionPositions[(sampleIdx - deviceFirstSample)], 1);
    //   }
    // } else {
    //   insertionPos = step;
    // }

    // // if (insertionPo

    // // if (insertionPos < finalSampleSize) {
    // //   printf("insertionPos %d finalSampleSize %d\n", insertionPos, finalSampleSize);
    // // }
    // assert(finalSampleSize > 0);
    // if (insertionPos >= finalSampleSize) {
    //   printf("insertionPos %d finalSampleSize %ld sample %d\n", insertionPos, finalSampleSize, sampleIdx);
    // }
    // assert(insertionPos < finalSampleSize);
    // if (numberOfTransits<App>(step) == 1 and combineTwoSampleStores) {
    //   if (step % 2 == 1) {
    //     finalSamples[(sampleIdx - deviceFirstSample)*finalSampleSize + insertionPos - 1] = transit;
    //     if (isValidSampledVertex(neighbor, invalidVertex)) finalSamples[(sampleIdx - deviceFirstSample)*finalSampleSize + insertionPos] = neighbor;
    //   } else if (step == App().steps() - 1 && isValidSampledVertex(neighbor, invalidVertex)) {
    //     finalSamples[(sampleIdx - deviceFirstSample)*finalSampleSize + insertionPos] = neighbor;
    //   }
    // }
    // else {
    //   // if (STORE_TRANSIT_INDEX) {
    //   //   //Store Index of transit in each sample's output
    //   //   if (step == 0) {
    //   //     transitIndexInSample[threadId] = insertionPos;
    //   //   } else if (step != App().steps() - 1) {
    //   //     transitIndexInSample[threadId] = prevTransitIndexInSample[];
    //   //   }
    //   // }
    //   if (isValidSampledVertex(neighbor, invalidVertex))
    //     finalSamples[(sampleIdx - deviceFirstSample)*finalSampleSize + insertionPos] = neighbor;
    // }
  }
}

template<class SampleType, typename App, int THREADS, bool COALESCE_CURAND_LOAD, bool HAS_EXPLICIT_TRANSITS>
__global__ void identityKernel(const int step, const VertexID_t deviceFirstSample, const VertexID_t invalidVertex,
                               const VertexID_t* transitToSamplesKeys, const VertexID_t* transitToSamplesValues,
                               const size_t transitToSamplesSize, SampleType* samples, const size_t NumSamples,
                               VertexID_t* samplesToTransitKeys, VertexID_t* samplesToTransitValues,
                               VertexID_t* finalSamples, const size_t finalSampleSize, EdgePos_t* sampleInsertionPositions,
                               curandState* randStates, const int* kernelTypeForTransit, int numTransits) {
  __shared__ unsigned char shMemCuRand[sizeof(curandState)*THREADS];

  int threadId = threadIdx.x + blockDim.x * blockIdx.x;

  curandState* curandSrcPtr;

  if (COALESCE_CURAND_LOAD) {
    const int intsInRandState = sizeof(curandState)/sizeof(int);
    int* shStateBuff = (int*)&shMemCuRand[0];

    int* randStatesAsInts = (int*)randStates;
  
    for (int i = threadIdx.x; i < intsInRandState*blockDim.x; i += blockDim.x) {
      shStateBuff[i] = randStatesAsInts[i + blockDim.x*blockIdx.x];
    }

    __syncthreads();
    curandSrcPtr = (curandState*)(&shStateBuff[threadIdx.x*intsInRandState]);
  } else {
    curandSrcPtr = &randStates[threadId];
  }

  curandState* localRandState = curandSrcPtr;
  
  for (; threadId < transitToSamplesSize; threadId += gridDim.x * blockDim.x) {
    //__shared__ VertexID newNeigbhors[N_THREADS];
    EdgePos_t transitIdx;
    EdgePos_t transitNeighborIdx;
    VertexID_t transit;
    int kernelTy;
    bool continueExecution = true;

    continueExecution = threadId < transitToSamplesSize;
    
    int subWarpSize = subWarpSizeAtStep<App>(step);
    transitIdx = threadId/subWarpSize;
    transitNeighborIdx = threadId % subWarpSize;
    if (continueExecution && transitNeighborIdx == 0) {
      transit = transitToSamplesKeys[transitIdx];
      kernelTy = kernelTypeForTransit[transit];
    }

    transit = __shfl_sync(FULL_WARP_MASK, transit, 0, subWarpSize);
    kernelTy = __shfl_sync(FULL_WARP_MASK, kernelTy, 0, subWarpSize);

    continueExecution = continueExecution && transitNeighborIdx < App().stepSize(step);

    if ((useGridKernel && kernelTy == TransitKernelTypes::GridKernel && numTransits > 1) ||
        (useSubWarpKernel && kernelTy == TransitKernelTypes::SubWarpKernel && numTransits > 1) || 
        (useThreadBlockKernel && kernelTy == TransitKernelTypes::ThreadBlockKernel && numTransits > 1)) {
        continueExecution = false;
    }

    BCGPartition* bcg = (BCGPartition*)&bcgPartitionBuff[0];

    VertexID_t sampleIdx = -1;
    
    if (continueExecution && transitNeighborIdx == 0) {
      sampleIdx = transitToSamplesValues[transitIdx];
    }

    sampleIdx = __shfl_sync(FULL_WARP_MASK, sampleIdx, 0, subWarpSize);
    VertexID_t neighbor = invalidVertex;

    if (continueExecution and transit != invalidVertex) {

      BCGVertex bcgv(transit, bcg->graph, bcg->offset[transit]);
      EdgePos_t numTransitEdges = bcgv.outd;
      
      if (numTransitEdges != 0) {
        neighbor = App().next(step, &transit, sampleIdx, &samples[(sampleIdx - deviceFirstSample)], numTransitEdges, transitNeighborIdx, localRandState, &bcgv);
        // printf("Thread_%d %d -> %d\n", threadId, transit, neighbor);
      }
    }

    __syncwarp();
  
  if (continueExecution) {
    if (step != App().steps() - 1) {
      //No need to store at last step
      if (HAS_EXPLICIT_TRANSITS) {
        VertexID_t newTransit = App().stepTransits(step + 1, sampleIdx, samples[sampleIdx - deviceFirstSample], transitIdx, localRandState);
        samplesToTransitValues[threadId] = newTransit != -1 ? newTransit : invalidVertex;
      } else {
        samplesToTransitValues[threadId] = neighbor != -1 ? neighbor : invalidVertex;;
      }
      samplesToTransitKeys[threadId] = sampleIdx;
    }
  }

  // __syncwarp();
  // //FIXME: in deepwalk if there is an invalid vertex at step k, it will not store the
  // //transits of step k -1 due to coalescing the stores. 
  // EdgePos_t finalSampleSizeTillPreviousStep = 0;
  // EdgePos_t neighborsToSampleAtStep = 1;
  // EdgePos_t insertionPos = 0; 
  // if (numTransits > 1) {    
  //   if (step == 0) {
  //     insertionPos = transitNeighborIdx;
  //   } else {
  //     for (int _s = 0; _s < step; _s++) {
  //       neighborsToSampleAtStep *= App().stepSize(_s);
  //       finalSampleSizeTillPreviousStep += neighborsToSampleAtStep;
  //     }
  //     EdgePos_t insertionStartPosForTransit = 0;

  //     if (threadIdx.x % subWarpSize == 0) {
  //         insertionStartPosForTransit = utils::atomicAdd(&sampleInsertionPositions[sampleIdx - deviceFirstSample], App().stepSize(step));
  //     }
  //     insertionStartPosForTransit = __shfl_sync(FULL_WARP_MASK, insertionStartPosForTransit, 0, subWarpSize);
  //     insertionPos = finalSampleSizeTillPreviousStep + insertionStartPosForTransit + transitNeighborIdx;
  //   }
  // } else {
  //   insertionPos = step;
  // }

  // __syncwarp();

  // if (continueExecution) {
  //   if (combineTwoSampleStores && numTransits == 1) {
  //     //TODO: We can combine stores even when numberOfTransits<App>(step) > 1
  //     if (step % 2 == 1) {
  //       int2 *ptr = (int2*)&finalSamples[(sampleIdx - deviceFirstSample)*finalSampleSize + insertionPos - 1];
  //       int2 res;
  //       res.x = transit;
  //       res.y = neighbor;
  //       *ptr = res;
  //       //finalSamples[sample*finalSampleSize + insertionPos] = neighbor;
  //       // 修改：finalSamples长度为奇数导致misaligned address
  //     } else if (step == App().steps() - 1) {
  //       finalSamples[(sampleIdx - deviceFirstSample)*finalSampleSize + insertionPos] = neighbor;
  //     }
  //   } 
  //   else {
  //     if (isValidSampledVertex(neighbor, invalidVertex))
  //       finalSamples[(sampleIdx - deviceFirstSample)*finalSampleSize + insertionPos] = neighbor;
  //   }
  // }

  //TODO: We do not need atomic instead store indices of transit in another array,
  //wich can be accessed based on sample and transitIdx.
  }
}

template<class SampleType, typename App, int THREADS, int CACHE_SIZE, bool CACHE_EDGES, bool CACHE_WEIGHTS, int TRANSITS_PER_THREAD, bool ONDEMAND_CACHING, int STATIC_CACHE_SIZE, int SUB_WARP_SIZE>
__global__ void threadBlockKernel(const int step, const VertexID_t deviceFirstSample, 
                           const VertexID_t invalidVertex,
                           const VertexID_t* transitToSamplesKeys, const VertexID_t* transitToSamplesValues,
                           const size_t transitToSamplesSize, SampleType* samples, const size_t NumSamples,
                           VertexID_t* samplesToTransitKeys, VertexID_t* samplesToTransitValues,
                           VertexID_t* finalSamples, const size_t finalSampleSize, EdgePos_t* sampleInsertionPositions,
                           curandState* randStates, const int* kernelTypeForTransit, const VertexID_t* threadBlockKernelPositions, 
                           const EdgePos_t threadBlockKernelPositionsNum, int totalThreadBlocks,
                           int numTransitsAtStepPerSample, int finalSampleSizeTillPreviousStep) {
  #define EDGE_CACHE_SIZE (CACHE_EDGES ? CACHE_SIZE : 0)
  #define CURAND_SHMEM_SIZE (sizeof(curandState)*THREADS)
  #define NUM_THREAD_GROUPS (THREADS / LoadBalancing::LoadBalancingThreshold::BlockLevel)

  union unionShMem {
    struct {
      // unsigned char edgeAndWeightCache[EDGE_CACHE_SIZE*NUM_THREAD_GROUPS+WEIGHT_CACHE_SIZE*NUM_THREAD_GROUPS];
      // VertexID_t transitForSubWarp[NUM_THREAD_GROUPS];
      VertexID_t graphCache[EDGE_CACHE_SIZE*NUM_THREAD_GROUPS];
      EdgePos_t mapStartPos[NUM_THREAD_GROUPS][TRANSITS_PER_THREAD];
      EdgePos_t subWarpTransits[NUM_THREAD_GROUPS][TRANSITS_PER_THREAD][LoadBalancing::LoadBalancingThreshold::BlockLevel/SUB_WARP_SIZE];
      EdgePos_t subWarpSampleIdx[NUM_THREAD_GROUPS][TRANSITS_PER_THREAD][LoadBalancing::LoadBalancingThreshold::BlockLevel/SUB_WARP_SIZE];
      VertexID_t transitVertices[NUM_THREAD_GROUPS][TRANSITS_PER_THREAD];
      // unsigned char transitVertices[NUM_THREAD_GROUPS][TRANSITS_PER_THREAD*sizeof(CSR::Vertex)];
    };
    unsigned char shMemAlloc[sizeof(curandState)*THREADS];
  };

  typedef cub::WarpScan<int> WarpScan;
  __shared__ unionShMem shMem;
  __shared__ typename WarpScan::TempStorage temp_storage[BLOCK_DIM / WARP_LEN];

  int warp_id = threadIdx.x / WARP_LEN;
  int lane_id = threadIdx.x - warp_id * WARP_LEN;
    
  // CSR::Edge* edgesInShMem = CACHE_EDGES ? (CSR::Edge*)(&shMem.edgeAndWeightCache[0] + EDGE_CACHE_SIZE*(threadIdx.x/LoadBalancing::LoadBalancingThreshold::BlockLevel)) : nullptr;
  // float* edgeWeightsInShMem = CACHE_WEIGHTS ? (float*)&shMem.edgeAndWeightCache[EDGE_CACHE_SIZE] : nullptr;
  VertexID_t* graphInShMem = CACHE_EDGES ? shMem.graphCache + EDGE_CACHE_SIZE * (threadIdx.x / LoadBalancing::LoadBalancingThreshold::BlockLevel) : nullptr;
  
  const int stepSize = App().stepSize(step);
  curandState* curandSrcPtr;

  const int subWarpSize = SUB_WARP_SIZE;

  const int intsInRandState = sizeof(curandState)/sizeof(int);
  int* shStateBuff = (int*)&shMem.shMemAlloc[0];

  int* randStatesAsInts = (int*)randStates;
  
  //Load curand only for the number of threads that are going to do sampling in this warp
  for (int i = threadIdx.x; i < intsInRandState*(blockDim.x/subWarpSize)*stepSize; i += blockDim.x) {
    shStateBuff[i] = randStatesAsInts[i + blockDim.x*blockIdx.x];
  }

  __syncthreads();
  if (threadIdx.x % subWarpSize < stepSize) {
    //Load curand only for the threads that are going to do sampling.
    curandSrcPtr = (curandState*)(&shStateBuff[threadIdx.x*intsInRandState]);
  }

  curandState localRandState = (threadIdx.x % subWarpSize < stepSize)? *curandSrcPtr: curandState();
  
  BCGPartition* bcg = (BCGPartition*)&bcgPartitionBuff[0];

  for (int fullBlockIdx = blockIdx.x; fullBlockIdx < totalThreadBlocks; fullBlockIdx += gridDim.x) {
    EdgePos_t transitIdx = 0;
    static_assert(NUM_THREAD_GROUPS * TRANSITS_PER_THREAD <= THREADS);
    int fullWarpIdx = (threadIdx.x + fullBlockIdx * blockDim.x)/LoadBalancing::LoadBalancingThreshold::BlockLevel;

    if (threadIdx.x < NUM_THREAD_GROUPS * TRANSITS_PER_THREAD) {
      const int warpIdx = threadIdx.x/TRANSITS_PER_THREAD;
      const int transitIdx = threadIdx.x%TRANSITS_PER_THREAD;
      const int __fullWarpIdx = warpIdx + (fullBlockIdx * blockDim.x)/LoadBalancing::LoadBalancingThreshold::BlockLevel;

      if (TRANSITS_PER_THREAD * __fullWarpIdx + transitIdx < threadBlockKernelPositionsNum)
        shMem.mapStartPos[warpIdx][transitIdx] = threadBlockKernelPositions[TRANSITS_PER_THREAD * __fullWarpIdx + transitIdx];
      else
        shMem.mapStartPos[warpIdx][transitIdx] = -1;
    }
  
    __syncthreads();
    
    const int NUM_SUBWARPS_IN_TB = NUM_THREAD_GROUPS * (LoadBalancing::LoadBalancingThreshold::BlockLevel/SUB_WARP_SIZE);
    static_assert(NUM_SUBWARPS_IN_TB * TRANSITS_PER_THREAD <= THREADS);
    
    if (threadIdx.x < NUM_SUBWARPS_IN_TB * TRANSITS_PER_THREAD) {
      //Coalesce loads of transits per sub-warp by loading transits for all sub-warps in one warp.
      //FIXME: Fix this when SUB_WARP_SIZE < 32
      int subWarpIdx = threadIdx.x / TRANSITS_PER_THREAD;
      int transitI = threadIdx.x % TRANSITS_PER_THREAD;
      transitIdx = shMem.mapStartPos[subWarpIdx][transitI];
      //TODO: Specialize this for subWarpSize = 1.
      VertexID_t transit = invalidVertex;
      if (transitIdx != -1) {
        transit = transitToSamplesKeys[transitIdx];
        shMem.subWarpSampleIdx[subWarpIdx][transitI][0] = transitToSamplesValues[transitIdx];
      }
      shMem.subWarpTransits[subWarpIdx][transitI][0] = transit;
    }
    __syncthreads();

    if (threadIdx.x < NUM_SUBWARPS_IN_TB * TRANSITS_PER_THREAD) {
      //Load transit Vertex Object in a coalesced manner
      //TODO: Fix this for subwarpsize < 32
      int transitI = threadIdx.x % TRANSITS_PER_THREAD;
      int subWarpIdx = threadIdx.x / TRANSITS_PER_THREAD;
      VertexID transit = shMem.subWarpTransits[subWarpIdx][transitI][0];
      if (transit != invalidVertex)
        shMem.transitVertices[subWarpIdx][transitI] = transit;
    }
    __syncthreads();

    for (int transitI = 0; transitI < TRANSITS_PER_THREAD; transitI++) {
      int threadBlockWarpIdx = threadIdx.x / subWarpSize;
      //TODO: Support this for SubWarp != 32

      if (TRANSITS_PER_THREAD * fullWarpIdx + transitI >= threadBlockKernelPositionsNum) continue;

      __syncwarp(); //TODO: Add mask based on subwarp
      VertexID_t transit = -1;
      bool invalidateCache = false;
      if (threadIdx.x % subWarpSize == 0) {
        invalidateCache = shMem.subWarpTransits[threadBlockWarpIdx][transitI][0] != transit || transitI == 0;
      }
      
      invalidateCache = __shfl_sync(FULL_WARP_MASK, invalidateCache, 0, subWarpSize);

      transit = shMem.subWarpTransits[threadBlockWarpIdx][transitI][0];
      if (transit == invalidVertex) 
        continue;
      
      __syncwarp();
      
      VertexID_t shMemTransitVertex = shMem.transitVertices[threadBlockWarpIdx][transitI];

      GlobalDecoder gd(shMemTransitVertex, bcg->graph, bcg->offset[shMemTransitVertex]);
      EdgePos_t outd = gd.outd;
      gd.set_sm_num(CACHE_SIZE, graphInShMem);

      for (int this_warp = warp_id; this_warp < min(CACHE_SIZE, gd.block_num); this_warp += BLOCK_DIM / WARP_LEN) {
        VertexID_t this_len = gd.get_len(this_warp);
        
        if (this_len) {
          VertexID_t lst_v = gd.get_head(this_warp);
          gd.write_vertex(lst_v, this_warp, 0);
          --this_len;

          for (int thread_iter = 0; thread_iter < this_len; thread_iter += WARP_LEN) {
            int this_thread = thread_iter + lane_id;
            VertexID_t sum_v;
            VertexID_t body_v = 0;
            if (this_thread < this_len) body_v = gd.get_body(this_thread);
            WarpScan(temp_storage[warp_id]).InclusiveSum(body_v, body_v, sum_v);
            __syncwarp();
            body_v += lst_v;
            
            if (this_thread < this_len) {
              gd.write_vertex(body_v, this_warp, this_thread + 1);
            }
            lst_v += sum_v;
          }
        }
      }
      __syncthreads();

      bool continueExecution = true;
      
      if (subWarpSize == 32) {
        assert(transit == shMemTransitVertex);
        //A thread will run next only when it's transit is same as transit of the threadblock.
        transitIdx = shMem.mapStartPos[threadBlockWarpIdx][transitI] + threadIdx.x/subWarpSize; //threadId/stepSize(step);
        VertexID_t transitNeighborIdx = threadIdx.x % subWarpSize;
        VertexID_t sampleIdx = shMem.subWarpSampleIdx[threadBlockWarpIdx][transitI][0];

        continueExecution = (transitNeighborIdx < stepSize); 

        VertexID_t neighbor = invalidVertex;
        if (outd > 0 && continueExecution) {
          neighbor = App().next(step, &transit, sampleIdx, &samples[sampleIdx-deviceFirstSample], outd, transitNeighborIdx, &localRandState, &gd);
        }

        if (continueExecution) {
          if (step != App().steps() - 1) {
            //No need to store at last step
            samplesToTransitKeys[transitIdx] = sampleIdx; //TODO: Update this for khop to transitIdx + transitNeighborIdx
            if (App().hasExplicitTransits()) {
              VertexID_t newTransit = App().stepTransits(step, sampleIdx, samples[sampleIdx-deviceFirstSample], transitIdx, &localRandState);
              samplesToTransitValues[transitIdx] = newTransit != -1 ? newTransit : invalidVertex;
            } else {
              samplesToTransitValues[transitIdx] = neighbor != -1 ? neighbor : invalidVertex;
            }
          }
        }

        // EdgePos_t insertionPos = transitNeighborIdx; 
        // if (numTransitsAtStepPerSample > 1) {
        //   if (step == 0) insertionPos = transitNeighborIdx;
        //   else {             
        //     EdgePos_t insertionStartPosForTransit = 0;
        //     //FIXME: 
        //     if (isValidSampledVertex(neighbor, invalidVertex) && threadIdx.x % subWarpSize == 0) {
        //       insertionStartPosForTransit = utils::atomicAdd(&sampleInsertionPositions[sampleIdx-deviceFirstSample], stepSize);
        //     }
        //     insertionStartPosForTransit = __shfl_sync(FULL_WARP_MASK, insertionStartPosForTransit, 0, subWarpSize);
        //     insertionPos = finalSampleSizeTillPreviousStep + insertionStartPosForTransit + transitNeighborIdx;
        //   }
        // } else insertionPos = step;

        // if (continueExecution) assert(insertionPos < finalSampleSize);            

        // if (combineTwoSampleStores && numTransitsAtStepPerSample == 1) {
        //   if (step % 2 == 1) {
        //     int2 *ptr = (int2*)&finalSamples[(sampleIdx-deviceFirstSample)*finalSampleSize + insertionPos - 1];
        //     int2 res;
        //     res.x = transit;
        //     res.y = neighbor;
        //     *ptr = res;
        //     //finalSamples[sample*finalSampleSize + insertionPos] = neighbor;
        //   } else if (step == App().steps() - 1) {
        //     finalSamples[(sampleIdx-deviceFirstSample)*finalSampleSize + insertionPos] = neighbor;
        //   }
        // } else {
        //   if (continueExecution && isValidSampledVertex(neighbor, invalidVertex))
        //     finalSamples[(sampleIdx-deviceFirstSample)*finalSampleSize + insertionPos] = neighbor;
        // }
        //TODO: We do not need atomic instead store indices of transit in another array,
        //wich can be accessed based on sample and transitIdx.

      }
    }
  }
}

template<class SampleType, typename App, int THREADS, int CACHE_SIZE, bool CACHE_EDGES, bool CACHE_WEIGHTS, bool COALESCE_GL_LOADS, int TRANSITS_PER_THREAD, 
bool COALESCE_CURAND_LOAD, bool ONDEMAND_CACHING, int STATIC_CACHE_SIZE, int SUB_WARP_SIZE>
__global__ void gridKernel(const int step, const VertexID_t deviceFirstSample, 
                           const VertexID_t invalidVertex,
                           const VertexID_t* transitToSamplesKeys, const VertexID_t* transitToSamplesValues,
                           const size_t transitToSamplesSize, SampleType* samples, const size_t NumSamples,
                           VertexID_t* samplesToTransitKeys, VertexID_t* samplesToTransitValues,
                           VertexID_t* finalSamples, const size_t finalSampleSize, EdgePos_t* sampleInsertionPositions,
                           curandState* randStates, const int* kernelTypeForTransit, const VertexID_t* gridKernelTBPositions, 
                           const EdgePos_t gridKernelTBPositionsNum, int totalThreadBlocks, int numTransitsPerStepForSample,
                           int finalSampleSizeTillPreviousStep) {
  #define EDGE_CACHE_SIZE (CACHE_EDGES ? CACHE_SIZE : 0)
  #define WEIGHT_CACHE_SIZE (CACHE_WEIGHTS ? CACHE_SIZE*sizeof(float) : 0)
  #define CURAND_SHMEM_SIZE (sizeof(curandState)*THREADS)
  // #define COALESCE_GL_LOADS_SHMEM_SIZE ()

  union unionShMem {
    struct {
      // unsigned char edgeAndWeightCache[EDGE_CACHE_SIZE+WEIGHT_CACHE_SIZE];
      VertexID_t graphCache[EDGE_CACHE_SIZE];
      bool invalidateCache;
      VertexID_t transitForTB;
      EdgePos_t mapStartPos[TRANSITS_PER_THREAD];
      EdgePos_t subWarpTransits[TRANSITS_PER_THREAD][THREADS/SUB_WARP_SIZE];
      EdgePos_t subWarpSampleIdx[TRANSITS_PER_THREAD][THREADS/SUB_WARP_SIZE];
      VertexID_t transitVertices[TRANSITS_PER_THREAD];
      // unsigned char transitVertices[TRANSITS_PER_THREAD*sizeof(CSR::Vertex)];
    };
    unsigned char shMemAlloc[sizeof(curandState)*THREADS];
  };

  typedef cub::WarpScan<int> WarpScan;
  __shared__ unionShMem shMem;
  __shared__ typename WarpScan::TempStorage temp_storage[BLOCK_DIM / WARP_LEN];

  int threadId = threadIdx.x + blockDim.x * blockIdx.x;
  int warp_id = threadIdx.x / WARP_LEN;
  int lane_id = threadIdx.x - warp_id * WARP_LEN;
  
  //__shared__ bool globalLoadBV[COALESCE_GL_LOADS ? CACHE_SIZE : 1];
  
  // CSR::Edge* edgesInShMem = CACHE_EDGES ? (CSR::Edge*)&shMem.edgeAndWeightCache[0] : nullptr;
  // float* edgeWeightsInShMem = CACHE_WEIGHTS ? (float*)&shMem.edgeAndWeightCache[EDGE_CACHE_SIZE] : nullptr;
  VertexID_t* graphInShMem = CACHE_EDGES ? shMem.graphCache : nullptr;
  // Graph_t* graphInShMem = CACHE_EDGES ? shMem.graphCache + EDGE_CACHE_SIZE * (threadIdx.x / LoadBalancing::LoadBalancingThreshold::BlockLevel) : nullptr;
  
  curandState* curandSrcPtr;
  const int stepSize = App().stepSize(step);

  const int subWarpSize = SUB_WARP_SIZE;

  if (COALESCE_CURAND_LOAD) {
    const int intsInRandState = sizeof(curandState)/sizeof(int);
    int* shStateBuff = (int*)&shMem.shMemAlloc[0];

    int* randStatesAsInts = (int*)randStates;
    
    //Load curand only for the number of threads that are going to do sampling in this warp
    for (int i = threadIdx.x; i < intsInRandState*(blockDim.x/subWarpSize)*stepSize; i += blockDim.x) {
      shStateBuff[i] = randStatesAsInts[i + blockDim.x*blockIdx.x];
    }

    __syncthreads();
    if (threadIdx.x % subWarpSize < stepSize) {
      //Load curand only for the threads that are going to do sampling.
      // int ld = threadIdx.x - (threadIdx.x/subWarpSize)*(subWarpSize-stepSize);
      curandSrcPtr = (curandState*)(&shStateBuff[threadIdx.x*intsInRandState]);
    }
  } else {
    curandSrcPtr = &randStates[threadId];
  }

  curandState localRandState = (threadIdx.x % subWarpSize < stepSize)? *curandSrcPtr: curandState();
  //curand_init(threadId, 0,0, &localRandState);

  //__shared__ VertexID newNeigbhors[N_THREADS];
  //if (threadIdx.x == 0) printf("blockIdx.x %d\n", blockIdx.x);
  //shRandStates[threadIdx.x] = randStates[threadId];  
  //__syncthreads();
  
  BCGPartition* bcg = (BCGPartition*)&bcgPartitionBuff[0];

  for (int fullBlockIdx = blockIdx.x; fullBlockIdx < totalThreadBlocks; fullBlockIdx += gridDim.x) {
    EdgePos_t transitIdx = 0;
    if (threadIdx.x < TRANSITS_PER_THREAD) {
      if (TRANSITS_PER_THREAD * fullBlockIdx + threadIdx.x < gridKernelTBPositionsNum) {
        shMem.mapStartPos[threadIdx.x] = gridKernelTBPositions[TRANSITS_PER_THREAD * fullBlockIdx + threadIdx.x];
      } else {
        shMem.mapStartPos[threadIdx.x] = -1;
      }
    }
    
    __syncthreads();
    if (threadIdx.x < THREADS/SUB_WARP_SIZE * TRANSITS_PER_THREAD) {
      //Coalesce loads of transits per sub-warp by loading transits for all sub-warps in one warp.
      // Assign THREADS/SUB_WARP_SIZE threads to each Transit in TRANSITS_PER_THREAD
      // static_assert ((THREADS/SUB_WARP_SIZE * TRANSITS_PER_THREAD) < THREADS);
      int transitI = threadIdx.x / (THREADS/SUB_WARP_SIZE);// * TRANSITS_PER_THREAD);
      transitIdx = shMem.mapStartPos[transitI] + threadIdx.x % (THREADS/SUB_WARP_SIZE);
      // if (!(transitIdx >= 0 && transitIdx < 57863 * 10)) {
      //   printf("transitIdx %d shMem.mapStartPos[transitI] %d\n", transitIdx, shMem.mapStartPos[transitI]);
      // }
      //TODO: Specialize this for subWarpSizez = 1.
      VertexID_t transit = invalidVertex;
      if (transitIdx != -1) {
        transit = transitToSamplesKeys[transitIdx];
        shMem.subWarpSampleIdx[transitI][threadIdx.x%(THREADS/SUB_WARP_SIZE)] = transitToSamplesValues[transitIdx];
      }
      shMem.subWarpTransits[transitI][threadIdx.x%(THREADS/SUB_WARP_SIZE)] = transit;
    }

    __syncthreads();
    if (threadIdx.x < TRANSITS_PER_THREAD) {
      //Load Transit Vertex of first subwarp in a Coalesced manner
      VertexID transit = shMem.subWarpTransits[threadIdx.x][0];
      if (transit != invalidVertex)
        shMem.transitVertices[threadIdx.x] = transit;
    }
    __syncwarp();
    
    for (int transitI = 0; transitI < TRANSITS_PER_THREAD; transitI++) {
      if (TRANSITS_PER_THREAD * (fullBlockIdx) + transitI >= gridKernelTBPositionsNum) continue;
      __syncthreads();

      VertexID_t transit = shMem.subWarpTransits[transitI][threadIdx.x/subWarpSize];

      // assert(4847571 >= shMemTransitVertex);      

      if (threadIdx.x == 0) {
        shMem.invalidateCache = shMem.transitForTB != transit || transitI == 0;
        shMem.transitForTB = transit;
      }
      __syncthreads();

      VertexID_t shMemTransitVertex = shMem.transitForTB;
      GlobalDecoder gd(shMemTransitVertex, bcg->graph, bcg->offset[shMemTransitVertex]);
      EdgePos_t outd = gd.outd;
      gd.set_sm_num(CACHE_SIZE, graphInShMem);

      for (int this_warp = warp_id; this_warp < min(CACHE_SIZE, gd.block_num); this_warp += BLOCK_DIM / WARP_LEN) {
        VertexID_t this_len = gd.get_len(this_warp);
        
        if (this_len) {
          VertexID_t lst_v = gd.get_head(this_warp);
          gd.write_vertex(lst_v, this_warp, 0);
          --this_len;

          for (int thread_iter = 0; thread_iter < this_len; thread_iter += WARP_LEN) {
            int this_thread = thread_iter + lane_id;
            VertexID_t sum_v;
            VertexID_t body_v = 0;
            if (this_thread < this_len) body_v = gd.get_body(this_thread);
            WarpScan(temp_storage[warp_id]).InclusiveSum(body_v, body_v, sum_v);
            __syncwarp();
            body_v += lst_v;
            
            if (this_thread < this_len) {
              gd.write_vertex(body_v, this_warp, this_thread + 1);
            }
            lst_v += sum_v;
          }
        }
      }
      __syncthreads();

      bool continueExecution = true;
      if (transit == shMem.transitForTB) {
        // assert(transit == shMemTransitVertex);
        //A thread will run next only when it's transit is same as transit of the threadblock.
        transitIdx = shMem.mapStartPos[transitI] + threadIdx.x/subWarpSize; //threadId/stepSize(step);
        VertexID_t transitNeighborIdx = threadIdx.x % subWarpSize;
        VertexID_t sampleIdx = shMem.subWarpSampleIdx[transitI][threadIdx.x/subWarpSize];;
        // if (threadIdx.x % subWarpSize == 0) {
        //   printf("1271: sampleIdx %d transit %d transitForTB %d numEdgesInShMem %d threadIdx.x %d blockIdx.x %d fullBlockIdx %d\n", sampleIdx, transit, shMem.transitForTB, numEdgesInShMem, threadIdx.x, blockIdx.x, fullBlockIdx);
        // }
        // if (threadIdx.x % subWarpSize == 0) {
        //   sampleIdx = transitToSamplesValues[transitIdx];
        // }
        
        // sampleIdx = __shfl_sync(FULL_WARP_MASK, sampleIdx, 0, subWarpSize);

        continueExecution = (transitNeighborIdx < stepSize); 
        // if (threadIdx.x == 0 && kernelTypeForTransit[transit] != TransitKernelTypes::GridKernel) {
        //   printf("transit %d transitIdx %d gridDim.x %d\n", transit, transitIdx, gridDim.x);
        // }
        // assert (kernelTypeForTransit[transit] == TransitKernelTypes::GridKernel);

        VertexID_t neighbor = invalidVertex;
        // if (graph.device_csr->has_vertex(transit) == false)
        //   printf("transit %d\n", transit);
        
        if (outd > 0 && continueExecution)
          // neighbor = App().next(step, &transit, sampleIdx, numEdgesInShMem, transitNeighborIdx, &localRandState, &bcgv);
          // neighbor = 1;
          neighbor = App().next(step, &transit, sampleIdx, &samples[sampleIdx-deviceFirstSample], outd, transitNeighborIdx, &localRandState, &gd);
        // //EdgePos_t totalSizeOfSample = stepSizeAtStep<App>(step - 1);
        // if ((transit == 612657 || transit == 348930) && sampleIdx == 17175) {
        //   printf("transit %d fullBlockIdx  %d sampleIdx %d neighbor %d\n", transit, fullBlockIdx, sampleIdx, neighbor);
        // }
        
        if (continueExecution) {
          if (step != App().steps() - 1) {
            //No need to store at last step
            samplesToTransitKeys[transitIdx] = sampleIdx; //TODO: Update this for khop to transitIdx + transitNeighborIdx
            if (App().hasExplicitTransits()) {
              VertexID_t newTransit = App().stepTransits(step, sampleIdx, samples[sampleIdx-deviceFirstSample], transitIdx, &localRandState);
              samplesToTransitValues[transitIdx] = newTransit != -1 ? newTransit : invalidVertex;
            } else {
              samplesToTransitValues[transitIdx] = neighbor != -1 ? neighbor : invalidVertex;
            }
          }
        }

        // EdgePos_t insertionPos = transitNeighborIdx; 
        // if (numTransitsPerStepForSample > 1) {
        //   if (step == 0) {
        //     insertionPos = transitNeighborIdx;
        //   } else {
        //     EdgePos_t insertionStartPosForTransit = 0;
        //     //FIXME: 
        //     if (isValidSampledVertex(neighbor, invalidVertex) && threadIdx.x % subWarpSize == 0) {
        //       insertionStartPosForTransit = utils::atomicAdd(&sampleInsertionPositions[sampleIdx-deviceFirstSample], stepSize);
        //     }
        //     insertionStartPosForTransit = __shfl_sync(FULL_WARP_MASK, insertionStartPosForTransit, 0, subWarpSize);
        //     insertionPos = finalSampleSizeTillPreviousStep + insertionStartPosForTransit + transitNeighborIdx;
        //   }
        // } else {
        //   insertionPos = step;
        // }

        // if (continueExecution) {
        //   // if (insertionPos >= finalSampleSize) {
        //   //   printf("1353: sampleIdx %d insertionPos %d finalSampleSize %ld transit %d\n", sampleIdx, insertionPos, finalSampleSize, transit);
        //   // }
        //   assert(insertionPos < finalSampleSize);
        // }

        // if (combineTwoSampleStores && numTransitsPerStepForSample == 1) {
        //   if (step % 2 == 1) {
        //     int2 *ptr = (int2*)&finalSamples[(sampleIdx-deviceFirstSample)*finalSampleSize + insertionPos - 1];
        //     int2 res;
        //     res.x = transit;
        //     res.y = neighbor;
        //     *ptr = res;
        //     //finalSamples[sample*finalSampleSize + insertionPos] = neighbor;
        //   } else if (step == App().steps() - 1) {
        //     finalSamples[(sampleIdx-deviceFirstSample)*finalSampleSize + insertionPos] = neighbor;
        //   }
        // } else {
        //   if (continueExecution && isValidSampledVertex(neighbor, invalidVertex))
        //     finalSamples[(sampleIdx-deviceFirstSample)*finalSampleSize + insertionPos] = neighbor;
        // }
        //TODO: We do not need atomic instead store indices of transit in another array,
        //wich can be accessed based on sample and transitIdx.
      }

      __syncthreads();
    }
  }
}

template<typename App, int TB_THREADS, TransitKernelTypes kTy, bool WRITE_KERNELTYPES>
__global__ void partitionTransitsInKernels(int step, EdgePos_t* uniqueTransits, EdgePos_t* uniqueTransitCounts, 
                                           EdgePos_t* transitPositions,
                                           EdgePos_t uniqueTransitCountsNum, VertexID_t invalidVertex,
                                           EdgePos_t* gridKernelTransits, EdgePos_t* gridKernelTransitsNum,
                                           EdgePos_t* threadBlockKernelTransits, EdgePos_t* threadBlockKernelTransitsNum,
                                           EdgePos_t* subWarpKernelTransits, EdgePos_t* subWarpKernelTransitsNum,
                                           EdgePos_t* identityKernelTransits, EdgePos_t* identityKernelTransitsNum,
                                           int* kernelTypeForTransit, VertexID_t* transitToSamplesKeys) 
{
  //__shared__ EdgePos_t insertionPosOfThread[TB_THREADS];
  const int SHMEM_SIZE = 7*TB_THREADS;
  // __shared__ EdgePos_t trThreadBlocks[TB_THREADS];
  // __shared__ EdgePos_t trStartPos[TB_THREADS];
  typedef cub::BlockScan<int, TB_THREADS> BlockScan;
  __shared__ typename BlockScan::TempStorage temp_storage;
  //__shared__ typename BlockScan::TempStorage temp_storage2;
  __shared__ EdgePos_t shGridKernelTransits[SHMEM_SIZE];
  //__shared__ EdgePos_t warpsLastThreadVals;
  __shared__ EdgePos_t threadToTransitPrefixSum[TB_THREADS];
  __shared__ EdgePos_t threadToTransitPos[TB_THREADS];
  __shared__ VertexID_t threadToTransit[TB_THREADS];
  __shared__ EdgePos_t totalThreadGroups;
  __shared__ EdgePos_t threadGroupsInsertionPos;
//  __shared__ EdgePos_t gridKernelTransitsIter;

  int threadId = blockIdx.x * blockDim.x + threadIdx.x;

  if (threadIdx.x == 0) totalThreadGroups = 0;

  for (int i = threadIdx.x; i < SHMEM_SIZE; i += blockDim.x)
    shGridKernelTransits[i] = 0;

  __syncthreads();
  
  VertexID_t transit = uniqueTransits[threadId];
  EdgePos_t trCount = (threadId >= uniqueTransitCountsNum || transit == invalidVertex) ? -1: uniqueTransitCounts[threadId];
  EdgePos_t trPos = (threadId >= uniqueTransitCountsNum || transit == invalidVertex) ? -1: transitPositions[threadId];
  int subWarpSize = subWarpSizeAtStep<App>(step);
  
  int kernelType = -1;
  if (useGridKernel && trCount * subWarpSize >= LoadBalancing::LoadBalancingThreshold::GridLevel) {    
    kernelType = TransitKernelTypes::GridKernel;
  } else if (useThreadBlockKernel && trCount * subWarpSize >= LoadBalancing::LoadBalancingThreshold::BlockLevel) {
    kernelType = TransitKernelTypes::ThreadBlockKernel;
  } else if (useSubWarpKernel && trCount * subWarpSize >= LoadBalancing::LoadBalancingThreshold::SubWarpLevel) {
    kernelType = TransitKernelTypes::SubWarpKernel;
  } else {
    kernelType = TransitKernelTypes::IdentityKernel;
  }
  
  if (WRITE_KERNELTYPES) {
    if (threadId < uniqueTransitCountsNum && kernelType != IdentityKernel && transit != invalidVertex) {
      kernelTypeForTransit[transit] = kernelType;
    } 

    if (kernelType == IdentityKernel && transit != invalidVertex && trCount !=-1) {
      *identityKernelTransitsNum = 1;
    }
  }

  __syncthreads();

  // if (kernelType == TransitKernelTypes::ThreadBlockKernel && WRITE_KERNELTYPES && transit < 20000) {
  //   printf("1769: transit %d trCount %d trPos %d\n", transit, trCount, trPos);
  // }
  //TODO: Remove unnecessary "__syncthreads();" statements

  //for (int kTy = 1; kTy < TransitKernelTypes::SubWarpKernel; kTy++)
  EdgePos_t numThreadGroups = 0;
  EdgePos_t* glKernelTransitsNum, *glKernelTransits;
  const int threadGroupSize = (kTy == TransitKernelTypes::GridKernel) ? LoadBalancing::LoadBalancingThreshold::GridLevel/subWarpSize : 
                              ((kTy == TransitKernelTypes::ThreadBlockKernel) ? LoadBalancing::LoadBalancingThreshold::BlockLevel/subWarpSize : 
                              ((kTy == TransitKernelTypes::SubWarpKernel) ? LoadBalancing::LoadBalancingThreshold::SubWarpLevel : -1));

  if (kTy == TransitKernelTypes::GridKernel && useGridKernel) {
    if (kernelType == TransitKernelTypes::GridKernel) {
      numThreadGroups = DIVUP(trCount, threadGroupSize);
      threadToTransitPos[threadIdx.x] = trPos;
      threadToTransit[threadIdx.x] = transit;
    } else {
      numThreadGroups = 0;
      threadToTransitPos[threadIdx.x] = 0;
      threadToTransit[threadIdx.x] = -1;
    } 
    glKernelTransitsNum = gridKernelTransitsNum;
    glKernelTransits = gridKernelTransits;
  } else if (kTy == TransitKernelTypes::ThreadBlockKernel && useThreadBlockKernel) {
    if (kernelType == TransitKernelTypes::ThreadBlockKernel) {
      numThreadGroups = DIVUP(trCount, threadGroupSize);
      threadToTransitPos[threadIdx.x] = trPos;
      threadToTransit[threadIdx.x] = transit;
    } else {
      numThreadGroups = 0;
      threadToTransitPos[threadIdx.x] = 0;
      threadToTransit[threadIdx.x] = -1;
    }       
    glKernelTransitsNum = threadBlockKernelTransitsNum;
    glKernelTransits = threadBlockKernelTransits;
    // if (blockIdx.x == 0) {
    //   printf("threadIdx.x %d transit %d\n", threadIdx.x, transit);
    // }
  } else if (kTy == TransitKernelTypes::SubWarpKernel && useSubWarpKernel) {
    if (kernelType == TransitKernelTypes::SubWarpKernel) {
      numThreadGroups = DIVUP(trCount, threadGroupSize);
      threadToTransitPos[threadIdx.x] = trPos;
      threadToTransit[threadIdx.x] = transit;
    } else {
      numThreadGroups = 0;
      threadToTransitPos[threadIdx.x] = 0;
      threadToTransit[threadIdx.x] = -1;
    }       
    glKernelTransitsNum = subWarpKernelTransitsNum;
    glKernelTransits = subWarpKernelTransits;
  } else {
    return;
    // continue;
  }
  
  __syncthreads();
  //Get all grid kernel transits
  EdgePos_t prefixSumThreadData = 0;
  BlockScan(temp_storage).ExclusiveSum(numThreadGroups, prefixSumThreadData);
  
  __syncthreads();
  
  if (threadIdx.x == blockDim.x - 1) {
    totalThreadGroups = prefixSumThreadData + numThreadGroups;
    // if (kTy == 2 && blockIdx.x == 27) printf("totalThreadGroups %d kTy %d blockIdx.x %d\n", totalThreadGroups, kTy, blockIdx.x);
    threadGroupsInsertionPos = ::atomicAdd(glKernelTransitsNum, totalThreadGroups);
  }
  __syncthreads();

  threadToTransitPrefixSum[threadIdx.x] = prefixSumThreadData;
  
  __syncthreads();
  
  for (int tgIter = 0; tgIter < totalThreadGroups; tgIter += SHMEM_SIZE) {
    __syncthreads();
    for (int i = threadIdx.x; i < SHMEM_SIZE; i += blockDim.x) {
      shGridKernelTransits[i] = 0;
    }
  
    __syncthreads();
    
    int prefixSumIndex = prefixSumThreadData - tgIter;
    if (prefixSumIndex < 0 && prefixSumIndex + numThreadGroups > 0) {
      prefixSumIndex = 0;
    }
    if (numThreadGroups > 0) {
      if (prefixSumIndex >= 0 && prefixSumIndex < SHMEM_SIZE) {
        shGridKernelTransits[prefixSumIndex] = threadIdx.x;
      }
    }
    
    __syncthreads();

    for (int tbs = threadIdx.x; tbs < DIVUP(min(SHMEM_SIZE, totalThreadGroups - tgIter), TB_THREADS)*TB_THREADS; tbs += blockDim.x) {
      __syncthreads();
      int d = 0, e = 0;
      if (tbs < TB_THREADS) {
        d = (tbs < totalThreadGroups) ? shGridKernelTransits[tbs] : 0;
      } else if (threadIdx.x == 0) {
        d = (tbs < totalThreadGroups) ? max(shGridKernelTransits[tbs], shGridKernelTransits[tbs-1]): 0;
      } else {
        d = (tbs < totalThreadGroups) ? shGridKernelTransits[tbs] : 0;
      }
      
      __syncthreads();
      BlockScan(temp_storage).InclusiveScan(d, e, cub::Max());
      __syncthreads();

      if (tbs < totalThreadGroups) shGridKernelTransits[tbs] = e;

      __syncthreads();

      if (tbs + tgIter < totalThreadGroups) {
        EdgePos_t xx = shGridKernelTransits[tbs];
        assert(xx >= 0 && xx < blockDim.x);
        int previousTrPrefixSum = (tbs < totalThreadGroups && xx >= 0) ? threadToTransitPrefixSum[xx] : 0;

        EdgePos_t startPos = threadToTransitPos[xx];
        EdgePos_t pos = startPos + threadGroupSize*(tbs  + tgIter - previousTrPrefixSum);
        
        VertexID_t transit = threadToTransit[xx];
        if (transit != -1) {
          int idx = threadGroupsInsertionPos + tbs + tgIter;
          glKernelTransits[idx] = pos;
          assert(kernelTypeForTransit[transit] == kTy);
          assert(transitToSamplesKeys[pos] == transit);
        }
      }

      __syncthreads();
    }

    __syncthreads();
  }
  __syncthreads();
}

__global__ void invalidVertexStartPos(int step, VertexID_t* transitToSamplesKeys, size_t totalTransits, 
                                      const VertexID_t invalidVertex, EdgePos_t* outputStartPos)
{
  int threadId = threadIdx.x + blockIdx.x*blockDim.x;

  if (threadId >= totalTransits) {
    return;
  }

  //If first transit is invalid.
  if (threadId == 0) {
    if (transitToSamplesKeys[0] == invalidVertex) {
      *outputStartPos = 0;
    }
    // printf("outputStartPos %d\n", *outputStartPos);
    return;
  }

  //TODO: Optimize this using overlaped tilling
  if (transitToSamplesKeys[threadId - 1] != invalidVertex && 
      transitToSamplesKeys[threadId] == invalidVertex)
  {
    *outputStartPos = threadId;
    return;
      // printf("outputStartPos %d\n", *outputStartPos);
  }

  //If no transit is invalid 
  // if (threadId == totalTransits - 1) {
  //   printf("1666: threadIdx.x %d v %d invalidVertex %d\n", threadId, transitToSamplesKeys[threadId], invalidVertex);
  //   *outputStartPos = totalTransits - 1;
  // }
}

__global__ void init_curand_states(curandState* states, size_t num_states)
{
  int thread_id = blockIdx.x*blockDim.x + threadIdx.x;
  if (thread_id < num_states)
    curand_init(thread_id, threadIdx.x, 0, &states[thread_id]);
}

BCG* loadGraph(char* graph_file) {
  BCG* ret = nullptr;

  std::ifstream ifs;
  std::string file_path(graph_file);
  ifs.open(file_path + ".graph", std::ios::in | std::ios::binary | std::ios::ate);

  if (!ifs.is_open()) {
    std::cout << "Open graph file failed!" << std::endl;
    return nullptr;
  }

  std::streamsize size = ifs.tellg();
  ifs.seekg(0, std::ios::beg);

  std::vector<uint8_t> buffer(size);
  ifs.read((char*)buffer.data(), size);

  ret = new BCG();

  Graph_t tmp = 0;
  for (size_t i = 0; i < buffer.size(); i++) {
    tmp <<= 8;
    tmp += buffer[i];
    if ((i + 1) % GRAPH_BYTE == 0) {
      (ret->graph).push_back(tmp);
    }
  }

  if (size % GRAPH_BYTE) {
    int rem = size % GRAPH_BYTE;
    while (rem % GRAPH_BYTE) tmp <<= 8, rem++;
    (ret->graph).push_back(tmp);
  }
  ifs.close();

  // load offset
  
  // (ret->offset).push_back(0);
  // std::ifstream ifs_offset;
  // ifs_offset.open(file_path + ".offset", std::ios::in);
  // ifs_offset >> (ret->n_nodes);
  // Offset_t cur;
  // for (auto i = 0; i < (ret->n_nodes); i++) {
  //   ifs_offset >> cur;
  //   (ret->offset).push_back(cur);
  // }
  // ifs_offset.close();

  std::ifstream ifso;
  ifso.open(file_path + ".offset", std::ios::in | std::ios::binary | std::ios::ate);

  if (!ifso.is_open()) {
    std::cout << "Open offset file failed!" << std::endl;
    return nullptr;
  }

  size = ifso.tellg();
  ifso.seekg(0, std::ios::beg);

  buffer.clear();
  buffer.resize(size);

  ifso.read((char*)buffer.data(), size);

  ret->ubl = buffer[0];
  tmp = 0;
  for (size_t i = 1; i < buffer.size(); i++) {
    tmp <<= 8;
    tmp += buffer[i];
    if (i % GRAPH_BYTE == 0) {
      (ret->offset).push_back(tmp);
    }
  }

  --size;
  if (size % GRAPH_BYTE) {
    int rem = size % GRAPH_BYTE;
    while (rem % GRAPH_BYTE) tmp <<= 8, rem++;
    (ret->offset).push_back(tmp);
  }

  ret->n_nodes = (size << 3l) / (ret->ubl);

  ifso.close();

  printf("Graph loaded. Graph has %d vertices.\n", ret->n_nodes);

  return ret;
}

template<typename NextDoorData>
std::vector<GPUBCGPartition> transferToGPUs(NextDoorData& data,  BCG* bcg) {
  //Assume that whole graph can be stored in GPU Memory.
  //Hence, only one Graph Partition is created.
  
  std::vector<GPUBCGPartition> gpuBCGPartitions;
  //Copy full graph to GPU
  for (int device = 0; device < data.devices.size(); device++) {
    GPUBCGPartition gpuBCGPartition;
    CHK_CU(cudaSetDevice(data.devices[device]));
    BCGPartition deviceBCGPartition = copyPartitionToGPU(bcg, gpuBCGPartition);
    gpuBCGPartition.d_bcg = (BCGPartition*)bcgPartitionBuff;
    CHK_CU(cudaMemcpyToSymbol(bcgPartitionBuff, &deviceBCGPartition, sizeof(BCGPartition)));
    gpuBCGPartitions.push_back(gpuBCGPartition);
  }
  return gpuBCGPartitions;
}

template<typename App>
int getFinalSampleSize()
{
  size_t finalSampleSize = 0;
  size_t neighborsToSampleAtStep = 1;
  for (int step = 0; step < App().steps(); step++) {
    neighborsToSampleAtStep *= App().stepSize(step);
    finalSampleSize += neighborsToSampleAtStep;
  }

  return finalSampleSize;
}

template<typename SampleType, typename App>
bool allocNextDoorDataOnGPU(NextDoorData<SampleType, App>& data, const BCG* bcg) {
  char* deviceList;
  if ((deviceList = getenv("CUDA_DEVICES")) != nullptr) {
    std::string deviceListStr = deviceList;

    std::stringstream ss(deviceListStr);
    if (ss.peek() == '[')
        ss.ignore();
    for (int i; ss >> i;) {
      data.devices.push_back(i);    
      if (ss.peek() == ',')
        ss.ignore();
    }
    if (ss.peek() == ']')
      ss.ignore();
  } else {
    data.devices = {0};
  }

  std::cout << "Using GPUs: [";
  for (auto d : data.devices) {
    std::cout << d << ",";
  }
  std::cout << "]" << std::endl;

  // auto samples_num = data.n_nodes;
  auto samples_num = App().numSamples(data.n_nodes);

  for (int sampleIdx = 0; sampleIdx < samples_num; sampleIdx++) {
    SampleType sample = App().template initializeSample<SampleType>(sampleIdx, bcg);
    data.samples.push_back(sample);
    auto initialVertices = App().initialSample(sampleIdx, data.n_nodes, data.samples[data.samples.size() - 1]);
    if ((EdgePos_t)initialVertices.size() != App().initialSampleSize()) {
      //We require that number of vertices in sample initially are equal to the initialSampleSize
      printf ("initialSampleSize '%d' != initialSample(%d).size() '%ld'\n", App().initialSampleSize(), sampleIdx, initialVertices.size());
      abort();
    }

    data.initialContents.insert(data.initialContents.end(), initialVertices.begin(), initialVertices.end());
    for (auto v : initialVertices)
      data.initialTransitToSampleValues.push_back(sampleIdx);
  }

  //Size of each sample output
  size_t maxNeighborsToSample = App().initialSampleSize(); //TODO: Set initial vertices
  for (int step = 0; step < App().steps() - 1; step++) {
    maxNeighborsToSample *= App().stepSize(step);
  }

  int finalSampleSize = getFinalSampleSize<App>();
  std::cout << "Final Size of each sample: " << finalSampleSize << std::endl;
  std::cout << "Maximum Neighbors Sampled at each step: " << maxNeighborsToSample << std::endl;
  std::cout << "Number of Samples: " << samples_num << std::endl;
  data.INVALID_VERTEX = data.n_nodes;
  int maxBits = 0;
  while ((data.INVALID_VERTEX >> maxBits) != 0) {
    maxBits++;
  }
  
  data.maxBits = maxBits;
  
  //Allocate storage for final samples on GPU
  #ifdef COPY_BACK
  data.hFinalSamples = std::vector<VertexID_t>(finalSampleSize*samples_num);
  #endif
  data.dSamplesToTransitMapKeys = std::vector<VertexID_t*>(data.devices.size(), nullptr);
  data.dSamplesToTransitMapValues = std::vector<VertexID_t*>(data.devices.size(), nullptr);
  data.dTransitToSampleMapKeys = std::vector<VertexID_t*>(data.devices.size(), nullptr);
  data.dTransitToSampleMapValues = std::vector<VertexID_t*>(data.devices.size(), nullptr);
  data.dSampleInsertionPositions = std::vector<VertexID_t*>(data.devices.size(), nullptr);
  data.dNeighborhoodSizes = std::vector<EdgePos_t*>(data.devices.size(), nullptr);
  data.dCurandStates = std::vector<curandState*>(data.devices.size(), nullptr);
  data.maxThreadsPerKernel = std::vector<size_t>(data.devices.size(), 0);
  data.dFinalSamples = std::vector<VertexID_t*>(data.devices.size(), nullptr);
  data.dInitialSamples = std::vector<VertexID_t*>(data.devices.size(), nullptr);
  data.dOutputSamples = std::vector<SampleType*>(data.devices.size(), nullptr);
  const size_t numDevices = data.devices.size();
  for(auto deviceIdx = 0; deviceIdx < data.devices.size(); deviceIdx++) {
    auto device = data.devices[deviceIdx];
    //Per Device Allocation
    CHK_CU(cudaSetDevice(device));
    
    const size_t perDeviceNumSamples = PartDivisionSize(samples_num, deviceIdx, numDevices);
    const size_t deviceSampleStartPtr = PartStartPointer(samples_num, deviceIdx, numDevices);

    //Allocate storage and copy initial samples on GPU
    size_t partDivisionSize = App().initialSampleSize()*perDeviceNumSamples;
    size_t partStartPtr = App().initialSampleSize()*deviceSampleStartPtr;
    CHK_CU(cudaMalloc(&data.dInitialSamples[deviceIdx], sizeof(VertexID_t)*partDivisionSize));
    CHK_CU(cudaMemcpy(data.dInitialSamples[deviceIdx], &data.initialContents[0] + partStartPtr, 
                      sizeof(VertexID_t)*partDivisionSize, cudaMemcpyHostToDevice));

    //Allocate storage for samples on GPU
    if (sizeof(SampleType) > 0) {
      CHK_CU(cudaMalloc(&data.dOutputSamples[deviceIdx], sizeof(SampleType)*perDeviceNumSamples));
      CHK_CU(cudaMemcpy(data.dOutputSamples[deviceIdx], &data.samples[0] + deviceSampleStartPtr, sizeof(SampleType)*perDeviceNumSamples, 
                        cudaMemcpyHostToDevice));
    }

    // CHK_CU(cudaMalloc(&data.dFinalSamples[deviceIdx], sizeof(VertexID_t)*finalSampleSize*perDeviceNumSamples));
    // gpu_memset(data.dFinalSamples[deviceIdx], data.INVALID_VERTEX, finalSampleSize*perDeviceNumSamples);
    
    //Samples to Transit Map
    CHK_CU(cudaMalloc(&data.dSamplesToTransitMapKeys[deviceIdx], sizeof(VertexID_t)*perDeviceNumSamples*maxNeighborsToSample));
    CHK_CU(cudaMalloc(&data.dSamplesToTransitMapValues[deviceIdx], sizeof(VertexID_t)*perDeviceNumSamples*maxNeighborsToSample));

    //Transit to Samples Map
    CHK_CU(cudaMalloc(&data.dTransitToSampleMapKeys[deviceIdx], sizeof(VertexID_t)*perDeviceNumSamples*maxNeighborsToSample));
    CHK_CU(cudaMalloc(&data.dTransitToSampleMapValues[deviceIdx], sizeof(VertexID_t)*perDeviceNumSamples*maxNeighborsToSample));

    //Same as initial values of samples for first iteration
    CHK_CU(cudaMemcpy(data.dTransitToSampleMapKeys[deviceIdx], &data.initialContents[0] + partStartPtr, sizeof(VertexID_t)*partDivisionSize, 
                      cudaMemcpyHostToDevice));
    CHK_CU(cudaMemcpy(data.dTransitToSampleMapValues[deviceIdx], &data.initialTransitToSampleValues[0] + partStartPtr, 
                      sizeof(VertexID_t)*partDivisionSize, cudaMemcpyHostToDevice));
    //Insertion positions per transit vertex for each sample
    CHK_CU(cudaMalloc(&data.dSampleInsertionPositions[deviceIdx], sizeof(EdgePos_t)*perDeviceNumSamples));

    size_t curandDataSize = maxNeighborsToSample*perDeviceNumSamples*sizeof(curandState);
    const size_t curandSizeLimit = 5L*1024L*1024L*sizeof(curandState);
    if (curandDataSize < curandSizeLimit) {
      int maxSubWarpSize = 0;
      for (int s = 0; s < App().steps(); s++) {
        maxSubWarpSize = max(maxSubWarpSize, subWarpSizeAtStep<App>(s));
      }
      //Maximum threads for a kernel should ensure that for a transit for a sample all needed
      //neighbors are sampled.
      assert(maxSubWarpSize != 0);
      data.maxThreadsPerKernel[deviceIdx] = ROUNDUP(maxNeighborsToSample*perDeviceNumSamples, maxSubWarpSize*N_THREADS);
      curandDataSize = data.maxThreadsPerKernel[deviceIdx] * sizeof(curandState);
    } else {
      data.maxThreadsPerKernel[deviceIdx] = curandSizeLimit/sizeof(curandState);
      curandDataSize = curandSizeLimit;
    }
    printf("Maximum Threads Per Kernel: %ld\n", data.maxThreadsPerKernel[deviceIdx]);
    CHK_CU(cudaMalloc(&data.dCurandStates[deviceIdx], curandDataSize));
    init_curand_states<<<thread_block_size(data.maxThreadsPerKernel[deviceIdx], 256UL), 256UL>>> (data.dCurandStates[deviceIdx], data.maxThreadsPerKernel[deviceIdx]);
    CHK_CU(cudaDeviceSynchronize());
  }

  return true;
}

template<class SampleType, typename App>
void freeDeviceData(NextDoorData<SampleType, App>& data) 
{
  for(auto deviceIdx = 0; deviceIdx < data.devices.size(); deviceIdx++) {
    auto device = data.devices[deviceIdx];
    CHK_CU(cudaSetDevice(device));
    CHK_CU(cudaFree(data.dSamplesToTransitMapKeys[deviceIdx]));
    CHK_CU(cudaFree(data.dSamplesToTransitMapValues[deviceIdx]));
    CHK_CU(cudaFree(data.dTransitToSampleMapKeys[deviceIdx]));
    CHK_CU(cudaFree(data.dTransitToSampleMapValues[deviceIdx]));
    CHK_CU(cudaFree(data.dSampleInsertionPositions[deviceIdx]));
    CHK_CU(cudaFree(data.dCurandStates[deviceIdx]));
    // CHK_CU(cudaFree(data.dFinalSamples[deviceIdx]));
    CHK_CU(cudaFree(data.dInitialSamples[deviceIdx]));
    if (sizeof(SampleType) > 0) CHK_CU(cudaFree(data.dOutputSamples[deviceIdx]));
  }

  //TODO:
  for (int device = 0; device < data.devices.size(); device++) {
    CHK_CU(cudaSetDevice(data.devices[device]));
    CHK_CU(cudaFree(data.gpuBCGPartitions[device].d_offset));
    CHK_CU(cudaFree(data.gpuBCGPartitions[device].d_graph));
  }
}

template<class SampleType, typename App>
bool doTransitParallelSampling(NextDoorData<SampleType, App>& nextDoorData) {
  //Size of each sample output
  size_t maxNeighborsToSample = App().initialSampleSize();
  for (int step = 0; step < App().steps() - 1; step++) {
    maxNeighborsToSample *= App().stepSize(step);
  }

  const size_t numDevices = nextDoorData.devices.size();
  size_t finalSampleSize = getFinalSampleSize<App>();
  // printf("FSS %ld\n", finalSampleSize);

  for(auto deviceIdx = 0; deviceIdx < nextDoorData.devices.size(); deviceIdx++) {
    auto device = nextDoorData.devices[deviceIdx];
    CHK_CU(cudaSetDevice(device));
    const size_t perDeviceNumSamples = PartDivisionSize(nextDoorData.samples.size(), deviceIdx, numDevices);
    const size_t deviceSampleStartPtr = PartStartPointer(nextDoorData.samples.size(), deviceIdx, numDevices);
    if (App().steps() == 1) {
      CHK_CU(cudaMemcpy(nextDoorData.dSamplesToTransitMapValues[deviceIdx], 
                        &nextDoorData.initialContents[0] + App().initialSampleSize()*deviceSampleStartPtr, 
                        sizeof(VertexID_t)*App().initialSampleSize()*perDeviceNumSamples, 
                        cudaMemcpyHostToDevice));
      CHK_CU(cudaMemcpy(nextDoorData.dSamplesToTransitMapKeys[deviceIdx], 
                        &nextDoorData.initialTransitToSampleValues[0] + App().initialSampleSize()*deviceSampleStartPtr, 
                        sizeof(VertexID_t)*App().initialSampleSize()*perDeviceNumSamples, 
                        cudaMemcpyHostToDevice));
    } else {
      CHK_CU(cudaMemcpy(nextDoorData.dTransitToSampleMapKeys[deviceIdx], 
                        &nextDoorData.initialContents[0] + App().initialSampleSize()*deviceSampleStartPtr, 
                        sizeof(VertexID_t)*App().initialSampleSize()*perDeviceNumSamples, 
                        cudaMemcpyHostToDevice));
      CHK_CU(cudaMemcpy(nextDoorData.dTransitToSampleMapValues[deviceIdx], 
                        &nextDoorData.initialTransitToSampleValues[0] + App().initialSampleSize()*deviceSampleStartPtr,  
                        sizeof(VertexID_t)*App().initialSampleSize()*perDeviceNumSamples, 
                        cudaMemcpyHostToDevice));
    }
  }

  // for (auto v : nextDoorData.initialTransitToSampleValues) {
  //   if (v != 0) {
  //     printf("v %d\n", v);
  //   }
  // }
  std::vector<VertexID_t*> d_temp_storage = std::vector<VertexID_t*>(nextDoorData.devices.size());
  std::vector<size_t> temp_storage_bytes = std::vector<size_t>(nextDoorData.devices.size());

  std::vector<VertexID_t*> dUniqueTransits = std::vector<VertexID_t*>(nextDoorData.devices.size());
  std::vector<VertexID_t*> dUniqueTransitsCounts = std::vector<VertexID_t*>(nextDoorData.devices.size());
  std::vector<EdgePos_t*> dUniqueTransitsNumRuns = std::vector<EdgePos_t*>(nextDoorData.devices.size());
  std::vector<EdgePos_t*> dTransitPositions = std::vector<EdgePos_t*>(nextDoorData.devices.size());
  std::vector<EdgePos_t*> uniqueTransitNumRuns = std::vector<EdgePos_t*>(nextDoorData.devices.size());
   
  /**Pointers for each kernel type**/
  std::vector<EdgePos_t*> gridKernelTransitsNum = std::vector<EdgePos_t*>(nextDoorData.devices.size());
  std::vector<EdgePos_t*> dGridKernelTransitsNum = std::vector<EdgePos_t*>(nextDoorData.devices.size());
  std::vector<VertexID_t*> dGridKernelTransits = std::vector<VertexID_t*>(nextDoorData.devices.size());
  
  std::vector<EdgePos_t*> threadBlockKernelTransitsNum = std::vector<EdgePos_t*>(nextDoorData.devices.size());
  std::vector<EdgePos_t*> dThreadBlockKernelTransitsNum = std::vector<EdgePos_t*>(nextDoorData.devices.size());
  std::vector<VertexID_t*> dThreadBlockKernelTransits = std::vector<VertexID_t*>(nextDoorData.devices.size());

  std::vector<EdgePos_t*> subWarpKernelTransitsNum = std::vector<EdgePos_t*>(nextDoorData.devices.size());
  std::vector<EdgePos_t*> dSubWarpKernelTransitsNum = std::vector<EdgePos_t*>(nextDoorData.devices.size());
  std::vector<VertexID_t*> dSubWarpKernelTransits = std::vector<VertexID_t*>(nextDoorData.devices.size());

  std::vector<EdgePos_t*> identityKernelTransitsNum = std::vector<EdgePos_t*>(nextDoorData.devices.size());
  std::vector<EdgePos_t*> dIdentityKernelTransitsNum = std::vector<EdgePos_t*>(nextDoorData.devices.size());
  /**********************************/
  
  /****Variables for Collective Transit Sampling***/
  std::vector<EdgePos_t*> hSumNeighborhoodSizes = std::vector<EdgePos_t*>(nextDoorData.devices.size(), nullptr);
  std::vector<EdgePos_t*> dSumNeighborhoodSizes = std::vector<EdgePos_t*>(nextDoorData.devices.size(), nullptr);
  std::vector<EdgePos_t*> dSampleNeighborhoodPos = std::vector<EdgePos_t*>(nextDoorData.devices.size(), nullptr);
  std::vector<EdgePos_t*> dSampleNeighborhoodSizes = std::vector<EdgePos_t*>(nextDoorData.devices.size(), nullptr);
  std::vector<VertexID_t*> dCollectiveNeighborhoodCSRCols = std::vector<VertexID_t*>(nextDoorData.devices.size(), nullptr);
  std::vector<EdgePos_t*> dCollectiveNeighborhoodCSRRows = std::vector<EdgePos_t*>(nextDoorData.devices.size(), nullptr);

  std::vector<EdgePos_t*> dInvalidVertexStartPosInMap = std::vector<EdgePos_t*>(nextDoorData.devices.size(), nullptr);
  std::vector<EdgePos_t*> invalidVertexStartPosInMap = std::vector<EdgePos_t*>(nextDoorData.devices.size(), nullptr);
  
  /*Single Memory Location on both CPU and GPU for transferring
   *number of transits for all kernels */
  std::vector<EdgePos_t*> dKernelTransitNums = std::vector<EdgePos_t*>(nextDoorData.devices.size(), nullptr);
  std::vector<EdgePos_t*> hKernelTransitNums= std::vector<EdgePos_t*>(nextDoorData.devices.size(), nullptr);
  const int NUM_KERNEL_TYPES = TransitKernelTypes::NumKernelTypes + 1;

  std::vector<int*> dKernelTypeForTransit = std::vector<int*>(nextDoorData.devices.size(), nullptr);;

  for(auto deviceIdx = 0; deviceIdx < nextDoorData.devices.size(); deviceIdx++) {
    auto device = nextDoorData.devices[deviceIdx];
    const size_t perDeviceNumSamples = PartDivisionSize(nextDoorData.samples.size(), deviceIdx, numDevices);
    CHK_CU(cudaSetDevice(device));
    CHK_CU(cudaMallocHost(&uniqueTransitNumRuns[deviceIdx], sizeof(EdgePos_t)));
    CHK_CU(cudaMallocHost(&hKernelTransitNums[deviceIdx], NUM_KERNEL_TYPES * sizeof(EdgePos_t)));
    
    gridKernelTransitsNum[deviceIdx] = hKernelTransitNums[deviceIdx];
    threadBlockKernelTransitsNum[deviceIdx] = hKernelTransitNums[deviceIdx] + 1;
    subWarpKernelTransitsNum[deviceIdx] = hKernelTransitNums[deviceIdx] + 2;
    identityKernelTransitsNum[deviceIdx] = hKernelTransitNums[deviceIdx] + 3;
    invalidVertexStartPosInMap[deviceIdx] = hKernelTransitNums[deviceIdx] + 4;
    //threadBlockKernelTransitsNum = hKernelTransitNums[3];
    CHK_CU(cudaMalloc(&dKernelTypeForTransit[deviceIdx], sizeof(VertexID_t)*nextDoorData.n_nodes));
    CHK_CU(cudaMalloc(&dTransitPositions[deviceIdx], 
                      sizeof(VertexID_t)*nextDoorData.n_nodes));
    CHK_CU(cudaMalloc(&dGridKernelTransits[deviceIdx], 
                      sizeof(VertexID_t)*perDeviceNumSamples*maxNeighborsToSample));
    std::cout << "perDeviceNumSamples " << perDeviceNumSamples << " maxNeighborsToSample " << maxNeighborsToSample << std::endl;
    if (useThreadBlockKernel) {
      CHK_CU(cudaMalloc(&dThreadBlockKernelTransits[deviceIdx], 
                      sizeof(VertexID_t)*perDeviceNumSamples*maxNeighborsToSample));
    }

    if (useSubWarpKernel) {
      CHK_CU(cudaMalloc(&dSubWarpKernelTransits[deviceIdx],
                      sizeof(VertexID_t)*perDeviceNumSamples*maxNeighborsToSample));
    }

    CHK_CU(cudaMalloc(&dKernelTransitNums[deviceIdx], NUM_KERNEL_TYPES * sizeof(EdgePos_t)));
    CHK_CU(cudaMemset(dKernelTransitNums[deviceIdx], 0, NUM_KERNEL_TYPES * sizeof(EdgePos_t)));
    dGridKernelTransitsNum[deviceIdx] = dKernelTransitNums[deviceIdx];
    dThreadBlockKernelTransitsNum[deviceIdx] = dKernelTransitNums[deviceIdx] + 1;
    dSubWarpKernelTransitsNum[deviceIdx] = dKernelTransitNums[deviceIdx] + 2;
    dIdentityKernelTransitsNum[deviceIdx] = dKernelTransitNums[deviceIdx] + 3;
    dInvalidVertexStartPosInMap[deviceIdx] = dKernelTransitNums[deviceIdx] + 4;

    //Check if the space runs out.
    //TODO: Use DoubleBuffer version that requires O(P) space.
    cub::DeviceRadixSort::SortPairs(d_temp_storage[deviceIdx], temp_storage_bytes[deviceIdx], 
              nextDoorData.dSamplesToTransitMapValues[deviceIdx], nextDoorData.dTransitToSampleMapKeys[deviceIdx], 
              nextDoorData.dSamplesToTransitMapKeys[deviceIdx], nextDoorData.dTransitToSampleMapValues[deviceIdx], 
              perDeviceNumSamples*maxNeighborsToSample);

    CHK_CU(cudaMalloc(&d_temp_storage[deviceIdx], temp_storage_bytes[deviceIdx]));
    CHK_CU(cudaMemset(nextDoorData.dSampleInsertionPositions[deviceIdx], 0, sizeof(EdgePos_t)*perDeviceNumSamples));

    CHK_CU(cudaMalloc(&dUniqueTransits[deviceIdx], (nextDoorData.n_nodes + 1)*sizeof(VertexID_t)));
    CHK_CU(cudaMalloc(&dUniqueTransitsCounts[deviceIdx], (nextDoorData.n_nodes + 1)*sizeof(VertexID_t)));
    CHK_CU(cudaMalloc(&dUniqueTransitsNumRuns[deviceIdx], sizeof(size_t)));

    CHK_CU(cudaMemset(dUniqueTransitsCounts[deviceIdx], 0, (nextDoorData.n_nodes + 1)*sizeof(VertexID_t)));
    CHK_CU(cudaMemset(dUniqueTransitsNumRuns[deviceIdx], 0, sizeof(size_t)));
  }

  std::vector<VertexID_t*> hAllSamplesToTransitMapKeys;
  std::vector<VertexID_t*> hAllTransitToSampleMapValues;
  std::vector<size_t> totalTransits = std::vector<size_t>(nextDoorData.devices.size());

  double loadBalancingTime = 0;
  double inversionTime = 0;
  double gridKernelTime = 0;
  double subWarpKernelTime = 0;
  double identityKernelTime = 0;
  double threadBlockKernelTime = 0;
  size_t neighborsToSampleAtStep = App().initialSampleSize();

  double end_to_end_t1 = convertTimeValToDouble(getTimeOfDay ());
  for (int step = 0; step < App().steps(); step++) {
    // printf("step %d start\n", step);
    const size_t numTransits = neighborsToSampleAtStep;
    std::vector<size_t> totalThreads = std::vector<size_t>(nextDoorData.devices.size());
    for(int i = 0; i < nextDoorData.devices.size(); i++) {
      const size_t perDeviceNumSamples = PartDivisionSize(nextDoorData.samples.size(), i, numDevices);
      totalThreads[i] = perDeviceNumSamples*neighborsToSampleAtStep;
    }
    // std::cout << "step " << step << std::endl;
    if (App().steps() == 1) {
      //FIXME: Currently a non-sorted Transit to Sample Map is passed to both TP and TP+LB.
      //Here, if there is only one step, a sorted map is passed.
      //Fix this to make sure a sorted map is always passed.
      double inversionT1 = convertTimeValToDouble(getTimeOfDay ());
      for(auto deviceIdx = 0; deviceIdx < nextDoorData.devices.size(); deviceIdx++) {
        auto device = nextDoorData.devices[deviceIdx];
        CHK_CU(cudaSetDevice(device));
        //Invert sample->transit map by sorting samples based on the transit vertices
        cub::DeviceRadixSort::SortPairs(d_temp_storage[deviceIdx], temp_storage_bytes[deviceIdx], 
                                        nextDoorData.dSamplesToTransitMapValues[deviceIdx], nextDoorData.dTransitToSampleMapKeys[deviceIdx], 
                                        nextDoorData.dSamplesToTransitMapKeys[deviceIdx], nextDoorData.dTransitToSampleMapValues[deviceIdx], 
                                        totalThreads[deviceIdx], 0, nextDoorData.maxBits);
        CHK_CU(cudaGetLastError());
      }
      
      CUDA_SYNC_DEVICE_ALL(nextDoorData);
      double inversionT2 = convertTimeValToDouble(getTimeOfDay ());
      //std::cout << "inversionTime at step " << step << " : " << (inversionT2 - inversionT1) << std::endl; 
      inversionTime += (inversionT2 - inversionT1);
    }

    neighborsToSampleAtStep = neighborsToSampleAtStep * App().stepSize(step);    
    for(int i = 0; i < nextDoorData.devices.size(); i++) {
      const size_t perDeviceNumSamples = PartDivisionSize(nextDoorData.samples.size(), i, numDevices);
      totalThreads[i] = perDeviceNumSamples*neighborsToSampleAtStep;
    }

    if (step == 0 && App().steps() > 1) {
      //When not doing load balancing call baseline transit parallel
      for(auto deviceIdx = 0; deviceIdx < nextDoorData.devices.size(); deviceIdx++) {
        auto device = nextDoorData.devices[deviceIdx];
        CHK_CU(cudaSetDevice(device));
        const VertexID_t deviceSampleStartPtr = PartStartPointer(nextDoorData.samples.size(), deviceIdx, numDevices);
        for (int threadsExecuted = 0; threadsExecuted < totalThreads[deviceIdx]; threadsExecuted += nextDoorData.maxThreadsPerKernel[deviceIdx]) {
          size_t currExecutionThreads = min((size_t)nextDoorData.maxThreadsPerKernel[deviceIdx], totalThreads[deviceIdx] - threadsExecuted);
          samplingKernel<SampleType, App, TransitParallelMode::NextFuncExecution, 0><<<thread_block_size(currExecutionThreads, N_THREADS), N_THREADS>>>(step, 
                          threadsExecuted, currExecutionThreads, deviceSampleStartPtr, nextDoorData.INVALID_VERTEX,
                          (const VertexID_t*)nextDoorData.dTransitToSampleMapKeys[deviceIdx], (const VertexID_t*)nextDoorData.dTransitToSampleMapValues[deviceIdx],
                          totalThreads[deviceIdx], nextDoorData.dOutputSamples[deviceIdx], nextDoorData.samples.size(),
                          nextDoorData.dSamplesToTransitMapKeys[deviceIdx], nextDoorData.dSamplesToTransitMapValues[deviceIdx],
                          nextDoorData.dFinalSamples[deviceIdx], finalSampleSize, nextDoorData.dSampleInsertionPositions[deviceIdx],
                          nullptr,  nullptr,  nullptr,  nullptr, nextDoorData.dCurandStates[deviceIdx]);
          CHK_CU(cudaGetLastError());
        }
      }
      CUDA_SYNC_DEVICE_ALL(nextDoorData);
    } else {
      double loadBalancingT1 = convertTimeValToDouble(getTimeOfDay ());
      
      for(auto deviceIdx = 0; deviceIdx < nextDoorData.devices.size(); deviceIdx++) {
        auto device = nextDoorData.devices[deviceIdx];
        CHK_CU(cudaSetDevice(device));
        CHK_CU(cudaMemset(dKernelTransitNums[deviceIdx], 0, NUM_KERNEL_TYPES * sizeof(EdgePos_t)));
        CHK_CU(cudaMemset(dInvalidVertexStartPosInMap[deviceIdx], 0xFF, sizeof(EdgePos_t)));
        const size_t perDeviceNumSamples = PartDivisionSize(nextDoorData.samples.size(), deviceIdx, numDevices);
        totalTransits[deviceIdx] = perDeviceNumSamples*numTransits;

        //Find the index of first invalid transit vertex. 
        invalidVertexStartPos<<<DIVUP(totalTransits[deviceIdx], 256), 256>>>(step, nextDoorData.dTransitToSampleMapKeys[deviceIdx], 
                                                                              totalTransits[deviceIdx], nextDoorData.INVALID_VERTEX, 
                                                                              dInvalidVertexStartPosInMap[deviceIdx]);
        CHK_CU(cudaGetLastError());
      }

      CUDA_SYNC_DEVICE_ALL(nextDoorData);

      for(auto deviceIdx = 0; deviceIdx < nextDoorData.devices.size(); deviceIdx++) {
        auto device = nextDoorData.devices[deviceIdx];
        CHK_CU(cudaSetDevice(device));
        CHK_CU(cudaMemcpy(invalidVertexStartPosInMap[deviceIdx], dInvalidVertexStartPosInMap[deviceIdx], 
                          1 * sizeof(EdgePos_t), cudaMemcpyDeviceToHost));
        //Now the number of threads launched are equal to number of valid transit vertices
        if (*invalidVertexStartPosInMap[deviceIdx] == 0xFFFFFFFF) {
          *invalidVertexStartPosInMap[deviceIdx] = totalTransits[deviceIdx];
        }
        totalThreads[deviceIdx] = *invalidVertexStartPosInMap[deviceIdx];
      }

      for(auto deviceIdx = 0; deviceIdx < nextDoorData.devices.size(); deviceIdx++) {
        auto device = nextDoorData.devices[deviceIdx];
        CHK_CU(cudaSetDevice(device));
        void* dRunLengthEncodeTmpStorage = nullptr;
        size_t dRunLengthEncodeTmpStorageSize = 0;
        //Find the number of transit vertices
        cub::DeviceRunLengthEncode::Encode(dRunLengthEncodeTmpStorage, dRunLengthEncodeTmpStorageSize, 
                                          nextDoorData.dTransitToSampleMapKeys[deviceIdx],
                                          dUniqueTransits[deviceIdx], dUniqueTransitsCounts[deviceIdx], 
                                          dUniqueTransitsNumRuns[deviceIdx], totalThreads[deviceIdx]);

        if (dRunLengthEncodeTmpStorageSize > temp_storage_bytes[deviceIdx]) {
          temp_storage_bytes[deviceIdx] = dRunLengthEncodeTmpStorageSize;
          CHK_CU(cudaFree(d_temp_storage[deviceIdx]));
          CHK_CU(cudaMalloc(&d_temp_storage[deviceIdx], temp_storage_bytes[deviceIdx]));
        }
        assert(dRunLengthEncodeTmpStorageSize <= temp_storage_bytes[deviceIdx]);
        dRunLengthEncodeTmpStorage = d_temp_storage[deviceIdx];
        
        cub::DeviceRunLengthEncode::Encode(dRunLengthEncodeTmpStorage, dRunLengthEncodeTmpStorageSize, 
                                          nextDoorData.dTransitToSampleMapKeys[deviceIdx],
                                          dUniqueTransits[deviceIdx], dUniqueTransitsCounts[deviceIdx], 
                                          dUniqueTransitsNumRuns[deviceIdx], totalThreads[deviceIdx]);

        CHK_CU(cudaGetLastError());
      }
      
      CUDA_SYNC_DEVICE_ALL(nextDoorData);
      
      for(auto deviceIdx = 0; deviceIdx < nextDoorData.devices.size(); deviceIdx++) {
        auto device = nextDoorData.devices[deviceIdx];
        CHK_CU(cudaSetDevice(device));
        CHK_CU(cudaMemcpy(uniqueTransitNumRuns[deviceIdx], dUniqueTransitsNumRuns[deviceIdx], 
                          sizeof(*uniqueTransitNumRuns[deviceIdx]), cudaMemcpyDeviceToHost));
        void* dExclusiveSumTmpStorage = nullptr;
        size_t dExclusiveSumTmpStorageSize = 0;
        //Exclusive sum to obtain the start position of each transit (and its samples) in the map
        cub::DeviceScan::ExclusiveSum(dExclusiveSumTmpStorage, dExclusiveSumTmpStorageSize, dUniqueTransitsCounts[deviceIdx], 
                                      dTransitPositions[deviceIdx], *uniqueTransitNumRuns[deviceIdx]);

        if (dExclusiveSumTmpStorageSize > temp_storage_bytes[deviceIdx]) {
          temp_storage_bytes[deviceIdx] = dExclusiveSumTmpStorageSize;
          CHK_CU(cudaFree(d_temp_storage[deviceIdx]));
          CHK_CU(cudaMalloc(&d_temp_storage[deviceIdx], temp_storage_bytes[deviceIdx]));
        }
        assert(dExclusiveSumTmpStorageSize <= temp_storage_bytes[deviceIdx]);
        dExclusiveSumTmpStorage = d_temp_storage[deviceIdx];

        cub::DeviceScan::ExclusiveSum(dExclusiveSumTmpStorage, dExclusiveSumTmpStorageSize, dUniqueTransitsCounts[deviceIdx],
                                      dTransitPositions[deviceIdx], *uniqueTransitNumRuns[deviceIdx]);

        CHK_CU(cudaGetLastError());
      }

      CUDA_SYNC_DEVICE_ALL(nextDoorData);

      int subWarpSize = subWarpSizeAtStep<App>(step);

      // printKernelTypes<App>(step, csr, dUniqueTransits[0], dUniqueTransitsCounts[0], dUniqueTransitsNumRuns[0]);
      for(auto deviceIdx = 0; deviceIdx < nextDoorData.devices.size(); deviceIdx++) {
        auto device = nextDoorData.devices[deviceIdx];
        CHK_CU(cudaSetDevice(device));
        if (*uniqueTransitNumRuns[deviceIdx] == 0) 
          continue;

        partitionTransitsInKernels<App, 1024, TransitKernelTypes::GridKernel, true><<<thread_block_size((*uniqueTransitNumRuns[deviceIdx]), 1024), 1024>>>(step, dUniqueTransits[deviceIdx], dUniqueTransitsCounts[deviceIdx], 
            dTransitPositions[deviceIdx], *uniqueTransitNumRuns[deviceIdx], nextDoorData.INVALID_VERTEX, dGridKernelTransits[deviceIdx], dGridKernelTransitsNum[deviceIdx], 
            dThreadBlockKernelTransits[deviceIdx], dThreadBlockKernelTransitsNum[deviceIdx], dSubWarpKernelTransits[deviceIdx], dSubWarpKernelTransitsNum[deviceIdx], nullptr, 
            dIdentityKernelTransitsNum[deviceIdx], dKernelTypeForTransit[deviceIdx], nextDoorData.dTransitToSampleMapKeys[deviceIdx]);

        CHK_CU(cudaGetLastError());
      }

      CUDA_SYNC_DEVICE_ALL(nextDoorData);

      if (useThreadBlockKernel and subWarpSize > 1) {
        for(auto deviceIdx = 0; deviceIdx < nextDoorData.devices.size(); deviceIdx++) {
          auto device = nextDoorData.devices[deviceIdx];
          CHK_CU(cudaSetDevice(device));
          if (*uniqueTransitNumRuns[deviceIdx] == 0) 
            continue;

          partitionTransitsInKernels<App, 1024, TransitKernelTypes::ThreadBlockKernel, false><<<thread_block_size((*uniqueTransitNumRuns[deviceIdx]), 1024), 1024>>>(step, dUniqueTransits[deviceIdx], dUniqueTransitsCounts[deviceIdx], 
              dTransitPositions[deviceIdx], *uniqueTransitNumRuns[deviceIdx], nextDoorData.INVALID_VERTEX, dGridKernelTransits[deviceIdx], dGridKernelTransitsNum[deviceIdx], 
              dThreadBlockKernelTransits[deviceIdx], dThreadBlockKernelTransitsNum[deviceIdx], dSubWarpKernelTransits[deviceIdx], dSubWarpKernelTransitsNum[deviceIdx], nullptr, 
              dIdentityKernelTransitsNum[deviceIdx], dKernelTypeForTransit[deviceIdx], nextDoorData.dTransitToSampleMapKeys[deviceIdx]);

          CHK_CU(cudaGetLastError());
        }

        CUDA_SYNC_DEVICE_ALL(nextDoorData);
      }


      for(auto deviceIdx = 0; deviceIdx < nextDoorData.devices.size(); deviceIdx++) {
        auto device = nextDoorData.devices[deviceIdx];
        CHK_CU(cudaSetDevice(device));
        if (*uniqueTransitNumRuns[deviceIdx] == 0) 
          continue;
        CHK_CU(cudaMemcpy(hKernelTransitNums[deviceIdx], dKernelTransitNums[deviceIdx], NUM_KERNEL_TYPES * sizeof(EdgePos_t), cudaMemcpyDeviceToHost));
        
        //std::cout << "hInvalidVertexStartPosInMap " << *invalidVertexStartPosInMap << " step " << step << std::endl;
        // GPUUtils::printDeviceArray(dGridKernelTransits, *gridKernelTransitsNum, ',');
        // getchar();
        // std::cout << "SubWarpSize at step " << step << " " << subWarpSize << std::endl;
        //From each Transit we sample stepSize(step) vertices
        totalThreads[deviceIdx] =  totalThreads[deviceIdx] * subWarpSize;
      }

      double loadBalancingT2 = convertTimeValToDouble(getTimeOfDay ());
      loadBalancingTime += (loadBalancingT2 - loadBalancingT1);

      bool noTransitsForAllDevices = true;
      for(auto deviceIdx = 0; deviceIdx < nextDoorData.devices.size(); deviceIdx++) {
        auto device = nextDoorData.devices[deviceIdx];
        if (*uniqueTransitNumRuns[deviceIdx] > 0) {
          noTransitsForAllDevices = false;
        }
      }

      if (noTransitsForAllDevices) break; //End Sampling because no more transits exists  

      double identityKernelTimeT1 = convertTimeValToDouble(getTimeOfDay ());

      for(auto deviceIdx = 0; deviceIdx < nextDoorData.devices.size(); deviceIdx++) {
        auto device = nextDoorData.devices[deviceIdx];
        CHK_CU(cudaSetDevice(device));
        if (*uniqueTransitNumRuns[deviceIdx] == 0) 
          continue;
        const size_t maxThreadBlocksPerKernel = min(8192L, nextDoorData.maxThreadsPerKernel[deviceIdx]/256L);
        const VertexID_t deviceSampleStartPtr = PartStartPointer(nextDoorData.samples.size(), deviceIdx, numDevices);
        if (*identityKernelTransitsNum[deviceIdx] > 0) {
          CHK_CU(cudaGetLastError());
          if (App().hasExplicitTransits()) {
            identityKernel<SampleType, App, 256, true, true><<<maxThreadBlocksPerKernel, 256>>>(step, 
              deviceSampleStartPtr, nextDoorData.INVALID_VERTEX,
              (const VertexID_t*)nextDoorData.dTransitToSampleMapKeys[deviceIdx], (const VertexID_t*)nextDoorData.dTransitToSampleMapValues[deviceIdx],
              totalThreads[deviceIdx], nextDoorData.dOutputSamples[deviceIdx], nextDoorData.samples.size(),
              nextDoorData.dSamplesToTransitMapKeys[deviceIdx], nextDoorData.dSamplesToTransitMapValues[deviceIdx],
              nextDoorData.dFinalSamples[deviceIdx], finalSampleSize, nextDoorData.dSampleInsertionPositions[deviceIdx],
              nextDoorData.dCurandStates[deviceIdx], dKernelTypeForTransit[deviceIdx], numberOfTransits<App>(step));
          } else {
            identityKernel<SampleType, App, 256, true, false><<<maxThreadBlocksPerKernel, 256>>>(step, 
              deviceSampleStartPtr, nextDoorData.INVALID_VERTEX,
              (const VertexID_t*)nextDoorData.dTransitToSampleMapKeys[deviceIdx], (const VertexID_t*)nextDoorData.dTransitToSampleMapValues[deviceIdx],
              totalThreads[deviceIdx], nextDoorData.dOutputSamples[deviceIdx], nextDoorData.samples.size(),
              nextDoorData.dSamplesToTransitMapKeys[deviceIdx], nextDoorData.dSamplesToTransitMapValues[deviceIdx],
              nextDoorData.dFinalSamples[deviceIdx], finalSampleSize, nextDoorData.dSampleInsertionPositions[deviceIdx],
              nextDoorData.dCurandStates[deviceIdx], dKernelTypeForTransit[deviceIdx], numberOfTransits<App>(step));
          }
          CHK_CU(cudaGetLastError());
        }
      }

      CUDA_SYNC_DEVICE_ALL(nextDoorData);

      double identityKernelTimeT2 = convertTimeValToDouble(getTimeOfDay ());
      identityKernelTime += (identityKernelTimeT2 - identityKernelTimeT1);

      if (subWarpSize > 1) {
        EdgePos_t finalSampleSizeTillPreviousStep = 0;
        EdgePos_t neighborsToSampleAtStep = 1;
        for (int _s = 0; _s < step; _s++) {
          neighborsToSampleAtStep *= App().stepSize(_s);
          finalSampleSizeTillPreviousStep += neighborsToSampleAtStep;
        }

        double threadBlockKernelTimeT1 = convertTimeValToDouble(getTimeOfDay ());

        for(auto deviceIdx = 0; deviceIdx < nextDoorData.devices.size(); deviceIdx++) {
          auto device = nextDoorData.devices[deviceIdx];
          CHK_CU(cudaSetDevice(device));
          //Process more than one thread blocks positions written in dGridKernelTransits per thread block.
          //Processing more can improve the locality if thread blocks have common transits.
          const int perThreadSamplesForThreadBlockKernel = 8; // Works best for KHop
          const int tbSize = 256L;
          const size_t maxThreadBlocksPerKernel = min(4096L, nextDoorData.maxThreadsPerKernel[deviceIdx]/tbSize);
          const VertexID_t deviceSampleStartPtr = PartStartPointer(nextDoorData.samples.size(), deviceIdx, numDevices);
          const size_t threadBlocks = DIVUP(((*threadBlockKernelTransitsNum[deviceIdx] * LoadBalancing::LoadBalancingThreshold::BlockLevel)/tbSize), perThreadSamplesForThreadBlockKernel);
          if (useThreadBlockKernel && *threadBlockKernelTransitsNum[deviceIdx] > 0){// && numberOfTransits<App>(step) > 1) {
            //FIXME: A Bug in Grid Kernel prevents it from being used when numberOfTransits for a sample at step are 1.
            // for (int threadBlocksExecuted = 0; threadBlocksExecuted < threadBlocks; threadBlocksExecuted += nextDoorData.maxThreadsPerKernel/256) {
              const bool CACHE_EDGES = true;
              const bool CACHE_WEIGHTS = false;
              const int CACHE_SIZE = (CACHE_EDGES || CACHE_WEIGHTS) ? 384 : 0;
              // printf("device %d threadBlockKernelTransitsNum %d threadBlocks %d\n", device, *threadBlockKernelTransitsNum[deviceIdx], threadBlocks);
              switch (subWarpSizeAtStep<App>(step)) {
                case 32:
                  threadBlockKernel<SampleType,App,tbSize,CACHE_SIZE,CACHE_EDGES,CACHE_WEIGHTS,perThreadSamplesForThreadBlockKernel,false,0,32><<<maxThreadBlocksPerKernel, tbSize>>>(step,
                    deviceSampleStartPtr, nextDoorData.INVALID_VERTEX,
                    (const VertexID_t*)nextDoorData.dTransitToSampleMapKeys[deviceIdx], (const VertexID_t*)nextDoorData.dTransitToSampleMapValues[deviceIdx],
                    totalThreads[deviceIdx],  nextDoorData.dOutputSamples[deviceIdx], nextDoorData.samples.size(),
                    nextDoorData.dSamplesToTransitMapKeys[deviceIdx], nextDoorData.dSamplesToTransitMapValues[deviceIdx],
                    nextDoorData.dFinalSamples[deviceIdx], finalSampleSize, nextDoorData.dSampleInsertionPositions[deviceIdx],
                    nextDoorData.dCurandStates[deviceIdx], dKernelTypeForTransit[deviceIdx], dThreadBlockKernelTransits[deviceIdx], *threadBlockKernelTransitsNum[deviceIdx], threadBlocks,  numberOfTransits<App>(step), finalSampleSizeTillPreviousStep);
                    break;
              }
              CHK_CU(cudaGetLastError());
              // CHK_CU(cudaDeviceSynchronize());
            // }
          }
        }

        CUDA_SYNC_DEVICE_ALL(nextDoorData);
        double threadBlockKernelTimeT2 = convertTimeValToDouble(getTimeOfDay ());
        threadBlockKernelTime += (threadBlockKernelTimeT2 - threadBlockKernelTimeT1);
        double gridKernelTimeT1 = convertTimeValToDouble(getTimeOfDay ());

        for(auto deviceIdx = 0; deviceIdx < nextDoorData.devices.size(); deviceIdx++) {
          auto device = nextDoorData.devices[deviceIdx];
          CHK_CU(cudaSetDevice(device));
          //Process more than one thread blocks positions written in dGridKernelTransits per thread block.
          //Processing more can improve the locality if thread blocks have common transits.
          const int perThreadSamplesForGridKernel = 16; // Works best for KHop
          //const int perThreadSamplesForGridKernel = 8;
          
          const size_t maxThreadBlocksPerKernel = min(4096L, nextDoorData.maxThreadsPerKernel[deviceIdx]/256L);
          const VertexID_t deviceSampleStartPtr = PartStartPointer(nextDoorData.samples.size(), deviceIdx, numDevices);
          const size_t threadBlocks = DIVUP(*gridKernelTransitsNum[deviceIdx], perThreadSamplesForGridKernel);
          // printf("device %d gridTransitsNum %d threadBlocks %d\n", device, *gridKernelTransitsNum[deviceIdx], threadBlocks);

          if (useGridKernel && *gridKernelTransitsNum[deviceIdx] > 0){// && numberOfTransits<App>(step) > 1) {
            //FIXME: A Bug in Grid Kernel prevents it from being used when numberOfTransits for a sample at step are 1.
              const bool CACHE_EDGES = true;
              const bool CACHE_WEIGHTS = false;
              const int CACHE_SIZE = (CACHE_EDGES || CACHE_WEIGHTS) ? 3*1024-10 : 0;
            
              switch (subWarpSizeAtStep<App>(step)) {
                case 32:
                  gridKernel<SampleType,App,256,CACHE_SIZE,CACHE_EDGES,CACHE_WEIGHTS,false,perThreadSamplesForGridKernel,true,false,256,32><<<maxThreadBlocksPerKernel, 256>>>(step,
                    deviceSampleStartPtr, nextDoorData.INVALID_VERTEX,
                    (const VertexID_t*)nextDoorData.dTransitToSampleMapKeys[deviceIdx], (const VertexID_t*)nextDoorData.dTransitToSampleMapValues[deviceIdx],
                    totalThreads[deviceIdx],  nextDoorData.dOutputSamples[deviceIdx], nextDoorData.samples.size(),
                    nextDoorData.dSamplesToTransitMapKeys[deviceIdx], nextDoorData.dSamplesToTransitMapValues[deviceIdx],
                    nextDoorData.dFinalSamples[deviceIdx], finalSampleSize, nextDoorData.dSampleInsertionPositions[deviceIdx],
                    nextDoorData.dCurandStates[deviceIdx], dKernelTypeForTransit[deviceIdx], dGridKernelTransits[deviceIdx], 
                    *gridKernelTransitsNum[deviceIdx], threadBlocks,numberOfTransits<App>(step), finalSampleSizeTillPreviousStep);
                    break;
                case 16:
                  gridKernel<SampleType,App,256,CACHE_SIZE,CACHE_EDGES,CACHE_WEIGHTS,false,perThreadSamplesForGridKernel,true,true,256,16><<<maxThreadBlocksPerKernel, 256>>>(step,
                    deviceSampleStartPtr, nextDoorData.INVALID_VERTEX,
                    (const VertexID_t*)nextDoorData.dTransitToSampleMapKeys[deviceIdx], (const VertexID_t*)nextDoorData.dTransitToSampleMapValues[deviceIdx],
                    totalThreads[deviceIdx],  nextDoorData.dOutputSamples[deviceIdx], nextDoorData.samples.size(),
                    nextDoorData.dSamplesToTransitMapKeys[deviceIdx], nextDoorData.dSamplesToTransitMapValues[deviceIdx],
                    nextDoorData.dFinalSamples[deviceIdx], finalSampleSize, nextDoorData.dSampleInsertionPositions[deviceIdx],
                    nextDoorData.dCurandStates[deviceIdx], dKernelTypeForTransit[deviceIdx], dGridKernelTransits[deviceIdx], *gridKernelTransitsNum[deviceIdx], threadBlocks,numberOfTransits<App>(step), finalSampleSizeTillPreviousStep);
                    break;
                case 8:
                gridKernel<SampleType,App,256,CACHE_SIZE,CACHE_EDGES,CACHE_WEIGHTS,false,perThreadSamplesForGridKernel,true,true,256,8><<<maxThreadBlocksPerKernel, 256>>>(step,
                    deviceSampleStartPtr, nextDoorData.INVALID_VERTEX,
                    (const VertexID_t*)nextDoorData.dTransitToSampleMapKeys[deviceIdx], (const VertexID_t*)nextDoorData.dTransitToSampleMapValues[deviceIdx],
                    totalThreads[deviceIdx],  nextDoorData.dOutputSamples[deviceIdx], nextDoorData.samples.size(),
                    nextDoorData.dSamplesToTransitMapKeys[deviceIdx], nextDoorData.dSamplesToTransitMapValues[deviceIdx],
                    nextDoorData.dFinalSamples[deviceIdx], finalSampleSize, nextDoorData.dSampleInsertionPositions[deviceIdx],
                    nextDoorData.dCurandStates[deviceIdx], dKernelTypeForTransit[deviceIdx], dGridKernelTransits[deviceIdx], *gridKernelTransitsNum[deviceIdx], threadBlocks,numberOfTransits<App>(step), finalSampleSizeTillPreviousStep);
                  break;
                case 4:
                gridKernel<SampleType,App,256,CACHE_SIZE,CACHE_EDGES,CACHE_WEIGHTS,false,perThreadSamplesForGridKernel,true,true,256,4><<<maxThreadBlocksPerKernel, 256>>>(step,
                    deviceSampleStartPtr, nextDoorData.INVALID_VERTEX,
                    (const VertexID_t*)nextDoorData.dTransitToSampleMapKeys[deviceIdx], (const VertexID_t*)nextDoorData.dTransitToSampleMapValues[deviceIdx],
                    totalThreads[deviceIdx],  nextDoorData.dOutputSamples[deviceIdx], nextDoorData.samples.size(),
                    nextDoorData.dSamplesToTransitMapKeys[deviceIdx], nextDoorData.dSamplesToTransitMapValues[deviceIdx],
                    nextDoorData.dFinalSamples[deviceIdx], finalSampleSize, nextDoorData.dSampleInsertionPositions[deviceIdx],
                    nextDoorData.dCurandStates[deviceIdx], dKernelTypeForTransit[deviceIdx], dGridKernelTransits[deviceIdx], *gridKernelTransitsNum[deviceIdx], threadBlocks,numberOfTransits<App>(step), finalSampleSizeTillPreviousStep);
                  break;
                case 2:
                gridKernel<SampleType,App,256,CACHE_SIZE,CACHE_EDGES,CACHE_WEIGHTS,false,perThreadSamplesForGridKernel,true,true,256,2><<<maxThreadBlocksPerKernel, 256>>>(step,
                    deviceSampleStartPtr, nextDoorData.INVALID_VERTEX,
                    (const VertexID_t*)nextDoorData.dTransitToSampleMapKeys[deviceIdx], (const VertexID_t*)nextDoorData.dTransitToSampleMapValues[deviceIdx],
                    totalThreads[deviceIdx],  nextDoorData.dOutputSamples[deviceIdx], nextDoorData.samples.size(),
                    nextDoorData.dSamplesToTransitMapKeys[deviceIdx], nextDoorData.dSamplesToTransitMapValues[deviceIdx],
                    nextDoorData.dFinalSamples[deviceIdx], finalSampleSize, nextDoorData.dSampleInsertionPositions[deviceIdx],
                    nextDoorData.dCurandStates[deviceIdx], dKernelTypeForTransit[deviceIdx], dGridKernelTransits[deviceIdx], *gridKernelTransitsNum[deviceIdx], threadBlocks,numberOfTransits<App>(step), finalSampleSizeTillPreviousStep);
                  break;
                case 1:
                gridKernel<SampleType,App,256,CACHE_SIZE,CACHE_EDGES,CACHE_WEIGHTS,false,perThreadSamplesForGridKernel,true,true,256,1><<<maxThreadBlocksPerKernel, 256>>>(step,
                    deviceSampleStartPtr, nextDoorData.INVALID_VERTEX,
                    (const VertexID_t*)nextDoorData.dTransitToSampleMapKeys[deviceIdx], (const VertexID_t*)nextDoorData.dTransitToSampleMapValues[deviceIdx],
                    totalThreads[deviceIdx],  nextDoorData.dOutputSamples[deviceIdx], nextDoorData.samples.size(),
                    nextDoorData.dSamplesToTransitMapKeys[deviceIdx], nextDoorData.dSamplesToTransitMapValues[deviceIdx],
                    nextDoorData.dFinalSamples[deviceIdx], finalSampleSize, nextDoorData.dSampleInsertionPositions[deviceIdx],
                    nextDoorData.dCurandStates[deviceIdx], dKernelTypeForTransit[deviceIdx], dGridKernelTransits[deviceIdx], *gridKernelTransitsNum[deviceIdx], threadBlocks,numberOfTransits<App>(step), finalSampleSizeTillPreviousStep);
                  break;
                default:
                  //TODO: Add others
                    break;
              }
              CHK_CU(cudaGetLastError());
            // }
          }
        }

        CUDA_SYNC_DEVICE_ALL(nextDoorData);

        double gridKernelTimeT2 = convertTimeValToDouble(getTimeOfDay ());
        gridKernelTime += (gridKernelTimeT2 - gridKernelTimeT1);
      }
    
    }

    if (step != App().steps() - 1) {
      double inversionT1 = convertTimeValToDouble(getTimeOfDay ());
      for(auto deviceIdx = 0; deviceIdx < nextDoorData.devices.size(); deviceIdx++) {
        auto device = nextDoorData.devices[deviceIdx];
        CHK_CU(cudaSetDevice(device));
        //Invert sample->transit map by sorting samples based on the transit vertices
        cub::DeviceRadixSort::SortPairs(d_temp_storage[deviceIdx], temp_storage_bytes[deviceIdx], 
                                        nextDoorData.dSamplesToTransitMapValues[deviceIdx], nextDoorData.dTransitToSampleMapKeys[deviceIdx], 
                                        nextDoorData.dSamplesToTransitMapKeys[deviceIdx], nextDoorData.dTransitToSampleMapValues[deviceIdx], 
                                        totalThreads[deviceIdx], 0, nextDoorData.maxBits);
        CHK_CU(cudaGetLastError());
      }
      CUDA_SYNC_DEVICE_ALL(nextDoorData);
      double inversionT2 = convertTimeValToDouble(getTimeOfDay ());
      //std::cout << "inversionTime at step " << step << " : " << (inversionT2 - inversionT1) << std::endl; 
      inversionTime += (inversionT2 - inversionT1);
      #if 0
      VertexID_t* hTransitToSampleMapKeys = new VertexID_t[totalThreads[0]];
      VertexID_t* hTransitToSampleMapValues = new VertexID_t[totalThreads[0]];
      VertexID_t* hSampleToTransitMapKeys = new VertexID_t[totalThreads[0]];
      VertexID_t* hSampleToTransitMapValues = new VertexID_t[totalThreads[0]];

      
      CHK_CU(cudaMemcpy(hSampleToTransitMapKeys, nextDoorData.dSamplesToTransitMapKeys[0], 
        totalThreads[0]*sizeof(VertexID_t), cudaMemcpyDeviceToHost));
      CHK_CU(cudaMemcpy(hSampleToTransitMapValues, nextDoorData.dSamplesToTransitMapValues[0],
        totalThreads[0]*sizeof(VertexID_t), cudaMemcpyDeviceToHost));
      CHK_CU(cudaMemcpy(hTransitToSampleMapKeys, nextDoorData.dTransitToSampleMapKeys[0], 
                        totalThreads[0]*sizeof(VertexID_t), cudaMemcpyDeviceToHost));
      CHK_CU(cudaMemcpy(hTransitToSampleMapValues, nextDoorData.dTransitToSampleMapValues[0],
                        totalThreads[0]*sizeof(VertexID_t), cudaMemcpyDeviceToHost));
      hAllTransitToSampleMapValues.push_back(hTransitToSampleMapValues);
      hAllSamplesToTransitMapKeys.push_back(hSampleToTransitMapKeys);

      printKeyValuePairs(hTransitToSampleMapKeys, hTransitToSampleMapValues, totalThreads[0], ',');
      #endif
    }
  }

  double end_to_end_t2 = convertTimeValToDouble(getTimeOfDay ());

  std::cout << "Transit Parallel: End to end time " << (end_to_end_t2 - end_to_end_t1) << " secs" << std::endl;
  std::cout << "InversionTime: " << inversionTime <<", " << "LoadBalancingTime: " << loadBalancingTime << ", " << "GridKernelTime: " << gridKernelTime << ", ThreadBlockKernelTime: " << threadBlockKernelTime << ", SubWarpKernelTime: " << subWarpKernelTime << ", IdentityKernelTime: "<< identityKernelTime << std::endl;
  for(auto deviceIdx = 0; deviceIdx < nextDoorData.devices.size(); deviceIdx++) {
    auto device = nextDoorData.devices[deviceIdx];
    CHK_CU(cudaFree(d_temp_storage[deviceIdx]));

    CHK_CU(cudaFree(dUniqueTransits[deviceIdx]));
    CHK_CU(cudaFree(dUniqueTransitsCounts[deviceIdx]));
    CHK_CU(cudaFree(dUniqueTransitsNumRuns[deviceIdx]));
    CHK_CU(cudaFree(dKernelTypeForTransit[deviceIdx]));
    CHK_CU(cudaFree(dTransitPositions[deviceIdx]));
    CHK_CU(cudaFree(dGridKernelTransits[deviceIdx]));
    CHK_CU(cudaFree(dThreadBlockKernelTransits[deviceIdx]));
    CHK_CU(cudaFree(dSubWarpKernelTransits[deviceIdx]));
  }
  
  #if 0
  for (int s = 1; s < App().steps() - 2; s++) {
    std::unordered_set<VertexID_t> s1, s2, intersection;
    for (int i = 100000; i < 200000; i++) {
      VertexID_t v1 = hAllSamplesToTransitMapKeys[s+1][i];
      VertexID_t v2 = hAllTransitToSampleMapValues[s+2][i];
      //printf("v1 %d v2 %d\n", v1, v2);
      s1.insert(v1);
      s2.insert(v2);
    }
    
    for (auto e : s1) {
      if (s2.count(e) == 1) intersection.insert(e);
    }

    std::cout << "s: " << s << " intersection: " << intersection.size() << std::endl;
  }
  #endif
  return true;
}

template<class SampleType, typename App>
std::vector<VertexID_t>& getFinalSamples(NextDoorData<SampleType, App>& nextDoorData)
{
  for(auto deviceIdx = 0; deviceIdx < nextDoorData.devices.size(); deviceIdx++) {
    auto device = nextDoorData.devices[deviceIdx];
    //Per Device Allocation
    CHK_CU(cudaSetDevice(device));
    const size_t numSamples = nextDoorData.samples.size();
    const size_t finalSampleSize = getFinalSampleSize<App>();

    const size_t perDeviceNumSamples = PartDivisionSize(numSamples, deviceIdx, nextDoorData.devices.size());
    const size_t deviceSampleStartPtr = PartStartPointer(numSamples, deviceIdx, nextDoorData.devices.size());

    CHK_CU(cudaMemcpy(&nextDoorData.hFinalSamples[0] + finalSampleSize * deviceSampleStartPtr, nextDoorData.dFinalSamples[deviceIdx], 
                      sizeof(nextDoorData.hFinalSamples[0]) * finalSampleSize * perDeviceNumSamples, cudaMemcpyDeviceToHost));
    CHK_CU(cudaMemcpy(&nextDoorData.samples[0] + deviceSampleStartPtr, nextDoorData.dOutputSamples[deviceIdx], 
                      perDeviceNumSamples*sizeof(SampleType), cudaMemcpyDeviceToHost));
    // int i = 0;
    // printf("CHecking for invalidvertex %d\n", nextDoorData.INVALID_VERTEX);
    // for (auto v : nextDoorData.hFinalSamples) {
    //   if (v==nextDoorData.INVALID_VERTEX) {printf("i %d\n", i);break;}
    // i++;
    // }
  }
  return nextDoorData.hFinalSamples;
}

template<class SampleType, typename App>
bool nextdoor(const char* graph_file, const int nruns) {

  //Load Graph
  BCG *bcg;
  if ((bcg = loadGraph((char*)graph_file)) == nullptr) {
    return false;
  }

  NextDoorData<SampleType, App> nextDoorData;
  nextDoorData.n_nodes = (bcg->n_nodes);

  double alloc_t1 = convertTimeValToDouble(getTimeOfDay());
  allocNextDoorDataOnGPU<SampleType, App>(nextDoorData, bcg);
  double alloc_t2 = convertTimeValToDouble(getTimeOfDay());

  double transfer_e2e_t1 = convertTimeValToDouble(getTimeOfDay());
  double e2e_t1 = convertTimeValToDouble(getTimeOfDay());

  double transfer_t1 = convertTimeValToDouble(getTimeOfDay());
  nextDoorData.gpuBCGPartitions = transferToGPUs(nextDoorData, bcg);
  double transfer_t2 = convertTimeValToDouble(getTimeOfDay());

  double run_t1 = convertTimeValToDouble(getTimeOfDay());
  for (int i = 0; i < nruns; i++) {
    doTransitParallelSampling<SampleType, App>(nextDoorData);
  }
  CUDA_SYNC_DEVICE_ALL(nextDoorData);
  double run_t2 = convertTimeValToDouble(getTimeOfDay());

  double transfer_e2e_t2 = convertTimeValToDouble(getTimeOfDay());

#ifdef GET_SAMPLE
  getFinalSamples(nextDoorData);
#endif
  double e2e_t2 = convertTimeValToDouble(getTimeOfDay());

#ifdef GET_SAMPLE
  size_t maxNeighborsToSample = 1;
  for (int step = 0; step < App().steps(); step++) {
    maxNeighborsToSample *= App().stepSize(step);
  }

  size_t finalSampleSize = getFinalSampleSize<App>();
  
  size_t totalSampledVertices = 0;

  for (auto s : nextDoorData.hFinalSamples) {
    totalSampledVertices += (int)(s != nextDoorData.INVALID_VERTEX);
  }
  std::cout << "totalSampledVertices " << totalSampledVertices << std::endl;


  for (size_t s = 0; s < nextDoorData.hFinalSamples.size(); s += finalSampleSize) {
    std::cout << "Contents of sample " << s/finalSampleSize << " [";
    for(size_t v = s; v < s + finalSampleSize; v++)
      std::cout << nextDoorData.hFinalSamples[v] << ", ";
    std::cout << "]" << std::endl;
  }
#endif

  std::cout << "EndToEndTime: " << (e2e_t2 - e2e_t1) << std::endl;
  std::cout << "transferEndToEndTime: " << (transfer_e2e_t2 - transfer_e2e_t1) << std::endl;
  std::cout << "transferDataTime: " << transfer_t2 - transfer_t1 << std::endl;
  std::cout << "runSamplingTime: " << run_t2 - run_t1 << std::endl;

  freeDeviceData(nextDoorData);

  return true;
}
#endif
