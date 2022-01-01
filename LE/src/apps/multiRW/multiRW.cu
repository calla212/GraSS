#include <stdlib.h>

#define NUM_ROOT_VERTICES 80

#define VERTICES_PER_SAMPLE 1

class MultiRWSample
{
public:
  VertexID_t rootVertices[NUM_ROOT_VERTICES];
  VertexID_t lastRootIdx;
};

struct MultiRWApp {
  __host__ __device__ int steps() {return 100;}

  __host__ __device__ 
  int stepSize(int k) {
    return 1;
  }

  template<typename SampleType>
  __device__ inline
  VertexID next(const int step, VertexID* transit, VertexID sampleIdx, SampleType* sample, VertexID_t numEdges, EdgePos_t neighbrID, curandState* state, BCGVertex * bcgv)
  {
    if (numEdges == 0) {
      return -1;
    }
    if (numEdges == 1) {
      VertexID_t v = bcgv->get_vertex(0);
      if (step > 0) {
        sample->rootVertices[sample->lastRootIdx] = v;
      }
      return v;
    }
    
    EdgePos_t x = RandNumGen::rand_int(state, numEdges);
    VertexID_t v = bcgv->get_vertex(x);

    if (step > 0) {
      sample->rootVertices[sample->lastRootIdx] = v;
    }

    return v;
  }

  __host__ __device__ int samplingType()
  {
    return SamplingType::IndividualNeighborhood;
  }

  __host__ __device__ EdgePos_t numSamples(VertexID_t n_nodes) {
    return n_nodes;
  }

  template<class SampleType>
  __host__ std::vector<VertexID_t> initialSample(int sampleIdx, VertexID_t n_nodes, SampleType& sample) {
    std::vector<VertexID_t> initialValue;
    initialValue.push_back(sample.rootVertices[0]);
    return initialValue;
  }

  __host__ __device__ EdgePos_t initialSampleSize(void)
  {
    return 1;
  }


  __host__ __device__ bool hasExplicitTransits()
  {
    return true;
  }

  template<class SampleType>
  __device__ VertexID_t stepTransits(int step, const VertexID_t sampleID, SampleType& sample, int transitIdx, curandState* randState)
  {
    //Use rejection sampling to sample based on the degree of vertices.
    int x = RandNumGen::rand_int(randState, NUM_ROOT_VERTICES);
    //printf("x %d\n", x);
    sample.lastRootIdx = x;
    return sample.rootVertices[x];
  }

  template<class SampleType>
  __host__ SampleType initializeSample(const VertexID_t sampleID, const BCG* bcg) {
    SampleType sample;
    //printf("sample %d\n", sampleID);
    for (int i = 0; i < NUM_ROOT_VERTICES; i++) {
      sample.rootVertices[i] = rand() % (bcg->n_nodes);
      // if (sampleID + i < graph->get_n_vertices()) {
      //   sample.rootVertices[i] = sampleID + i;
      // } else {
      //   sample.rootVertices[i] = sampleID;
      // }
    }
    return sample;
  }
};
