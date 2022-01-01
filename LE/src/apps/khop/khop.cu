struct KHopApp {

  __host__ __device__ int steps() {return 2;}

  __host__ __device__ 
  int stepSize(int k) {
    return ((k == 0) ? 10 : 25);
  }

  template<typename SampleType>
  __device__ inline
  VertexID next(const int step, VertexID* transit, VertexID sampleIdx, SampleType* sample, VertexID_t numEdges, EdgePos_t neighbrID, curandState* state, BCGVertex * bcgv) {
    if (numEdges == 0) {
      return -1;
    }

    if (numEdges == 1) {
      return bcgv->get_vertex(0);
    }
    
    EdgePos_t x = RandNumGen::rand_int(state, numEdges);

    return bcgv->get_vertex(x);
  }

  __host__ __device__ int samplingType()
  {
    return SamplingType::IndividualNeighborhood;
  }

  #define VERTICES_PER_SAMPLE 1


  __host__ __device__ EdgePos_t numSamples(VertexID_t n_nodes) {
    return n_nodes;
    // return n_nodes / VERTICES_PER_SAMPLE;
  }

  template<class SampleType>
  __host__ std::vector<VertexID_t> initialSample(int sampleIdx, VertexID_t n_nodes, SampleType& sample) {
    std::vector<VertexID_t> initialValue;
    for (int i = 0; i < VERTICES_PER_SAMPLE; i++)
      initialValue.push_back(sampleIdx % n_nodes);
    return initialValue;
  }

  __host__ __device__ EdgePos_t initialSampleSize(void)
  {
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

  template<class SampleType>
  __host__ SampleType initializeSample(const VertexID_t sampleID, const BCG* bcg)
  {
    SampleType sample;
    return sample;
  }
};

class KHopSample
{

};