#include <stdlib.h>    

struct RandomWalkApp {
  __host__ __device__ int steps() {return 80;}

  __host__ __device__ 
  int stepSize(int k) {
    return 1;
  }

  __host__ __device__ int samplingType()
  {
    return SamplingType::IndividualNeighborhood;
  }

  #define VERTICES_PER_SAMPLE 1

  __host__ __device__ EdgePos_t numSamples(VertexID_t n_nodes) {
    // return n_nodes * 2000;
    return n_nodes;
    // return n_nodes < 256 * 1024 ? 100 * n_nodes : n_nodes;
  }


  template<class SampleType>
  __host__ std::vector<VertexID_t> initialSample(int sampleIdx, VertexID_t n_nodes, SampleType& sample) {
    std::vector<VertexID_t> initialValue;
    for (int i = 0; i < VERTICES_PER_SAMPLE; i++)
      initialValue.push_back(sampleIdx % n_nodes);
    return initialValue;
  }

  __host__ __device__ EdgePos_t initialSampleSize(void) {
    return VERTICES_PER_SAMPLE;
  }

  __host__ __device__ bool hasExplicitTransits()
  {
    return false;
  }

  template<class SampleType>
  __host__ __device__ VertexID_t stepTransits(int step, const VertexID_t sampleID, SampleType& sample, int transitIdx, curandState* randState)
  {
    return -1;
  }

  // template<class SampleType>
  // __host__ SampleType initializeSample(CSR* graph, const VertexID_t sampleID) {
  //   SampleType sample = SampleType();
  //   return sample;
  // }

  template<class SampleType>
  __host__ SampleType initializeSample(const VertexID_t sampleID, const BCG* bcg) {
    SampleType sample = SampleType();
    return sample;
  }
};


struct DeepWalkApp : public RandomWalkApp {
  template<typename SampleType>
  __device__ inline
  VertexID next(const int step, VertexID* transit, VertexID sampleIdx, SampleType* sample, VertexID_t numEdges, EdgePos_t neighbrID, curandState* state, BCGVertex * bcgv) {
    if (numEdges == 0) {
      return -1;
    }
    
    EdgePos_t x = RandNumGen::rand_int(state, numEdges);

    return bcgv->get_vertex(x);
    // auto ret = bcgv -> get_vertex(x);
    // assert(ret >= 0);
    // assert(ret <= 3072441);
    // // printf("%d -%d-> %d\n", *transit, x, ret);
    // return ret;
  }
};

class DummySample
{

};
