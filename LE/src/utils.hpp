#include <time.h>
#include <sys/time.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <iterator>
#include <cub/cub.cuh>
#include "bcg.cuh"

#ifndef __UTILS_HPP__
#define __UTILS_HPP__
#define DIVUP(x,y) (((x) + ((y) - 1))/(y))
#define ROUNDUP(x,y) (DIVUP(x,y)*(y))
#define FULL_WARP_MASK 0xffffffff
#define WARP_SIZE (32)
#define CHK_CU(x) if (utils::is_cuda_error (x, __LINE__) == true) {assert(false);}
#define MAX(x,y) (((x)<(y))?(y):(x))
#define MIN(x,y) (((x)>(y))?(y):(x))
const int GRAPH_BYTE = 4;

#define CUDA_SYNC_DEVICE_ALL(nextDoorData) {\
for(auto device : nextDoorData.devices) {\
  CHK_CU(cudaSetDevice(device));\
  CHK_CU(cudaDeviceSynchronize());\
}\
}

#define PartDivisionSize(Total,PartIdx,NumParts) (PartIdx < NumParts - 1) ? Total/NumParts : Total - (NumParts - 1)*(Total/NumParts)
#define PartStartPointer(Total,PartIdx,NumParts) PartIdx * (Total/NumParts)

namespace utils {
  template<typename F, typename T> 
  __host__ __device__ __forceinline__
  bool binarySearch (F& array, T x, int size) 
  {
    EdgePos_t l = 0;
    EdgePos_t r = size - 1;
    
    while (l <= r) {
      EdgePos_t m = l + (r-l)/2;
      if (array[m] == x)
        return true;
      
      if (array[m] < x)
        l = m + 1;
      else
        r = m - 1;
    }
    return false;
  }

  template<typename T> 
  __host__ __device__ __forceinline__
  bool linearSearch (const T* array, T x, int size) 
  {
    for (EdgePos_t i = 0; i < size; i++) {
      if (array[i] == x) {
        return true;
      }
    }

    return false;
  }

  __device__
  EdgePos_t atomicAdd (EdgePos_t* ptr, const EdgePos_t val)
  {
    if (sizeof(EdgePos_t) == 8) {
      return (EdgePos_t)::atomicAdd((unsigned long long*)ptr, (unsigned long long)val);
    } else {
      return (EdgePos_t)::atomicAdd((int*)ptr, (int)val);
    }
  }

  double convertTimeValToDouble (struct timeval _time)
  {
    return ((double)_time.tv_sec) + ((double)_time.tv_usec)/1000000.0f;
  }

  template<class T>
  void print_container(T const &s)
  {
      std::copy(s.begin(),
              s.end(),
              std::ostream_iterator<int>(std::cout, " "));
  }

  struct timeval getTimeOfDay ()
  {
    struct timeval _time;

    if (gettimeofday (&_time, NULL) == -1) {
      fprintf (stderr, "gettimeofday returned -1\n");
      perror ("");
      abort ();
    }

    return _time;
  }

  __device__ __host__ bool intervals_intersect (int x1, int x2, int y1, int y2)
  {
    return x1 <= y2 && y1 <= x2;
  }


  bool is_cuda_error (cudaError_t error, int line) 
  {
    //cudaError_t error = cudaGetLastError ();
    if (error != cudaSuccess) {
      const char* error_string = cudaGetErrorString (error);
      std::cout << "Cuda Error: " << error_string << " " << line <<
      std::endl;
      return true;
    }

    return false;
  }

  template<class T1, class T2>
  inline T1 next_multiple(const T1 val, const T2 divisor)
  {
    if (val%divisor == 0) return val;
    return (val/divisor + 1)*divisor;
  }

  template<class T>
  inline T thread_block_size(const T total, const T tb_size)
  {
    if (total%tb_size == 0)
      return total/tb_size;
    return total/tb_size +1;
  }

  template<class T1, class T2, class T3>
  inline void set_till_next_multiple(T1& val, const T2 divisor, T3* mem, 
                                     const T3 init_val)
  {
    while (val%divisor != 0)
      mem[val++] = -1;
  }

  //Kernel to memset global memory
  template <class T>
  __global__ void memset_kernel(T* mem, T val, size_t nelems)
  {
    int id = threadIdx.x + blockIdx.x*blockDim.x;

    if (id < nelems) {
      mem[id] = val;
    }
  }

  template <class T>
  void gpu_memset(T* mem, T val, size_t nelems)
  {
    const size_t threads = 256;
    const size_t blocks = thread_block_size(nelems, threads);

    memset_kernel<<<blocks, threads>>>(mem, val, nelems);
    CHK_CU(cudaDeviceSynchronize());
  }

  template<class T>
  class RangeIterator
  {
    public:
      class iterator 
      {
      private:
         T it;

      public: 
        iterator(T _it) : it(_it) {}
        iterator operator++() {it++; return *this;}
        iterator operator--() {it--; return *this;}
        T operator*() {return it;}

        bool operator==(const iterator& rhs) {return it == rhs.it;}
        bool operator!=(const iterator& rhs) {return it != rhs.it;}
      };

    private:
      T first;
      T last;

    public:
      RangeIterator(T _first, T _last) : first(_first), last(_last) {}

      iterator begin() const
      {
        return iterator(first);
      }

      iterator end() const
      {
        return iterator(last+1);
      }
  };

  template<class T>
  size_t sizeof_vector(const T& vec)
  {
    return vec.size()*sizeof(vec[0]);
  }

  enum StorageLocationType
  {
    Host,
    Device, 
  };

  template<class T>
  class Array
  {
    private:
      T* data_;
      
      StorageLocationType storageLocationType_;

      size_t nelems_;
    public:
      Array (StorageLocationType storageLocation, size_t nelems) : Array(nullptr, storageLocation, nelems)
      {
      }

      Array (T* ptr, StorageLocationType storageLocation, size_t nelems) : data_(ptr), storageLocationType_(storageLocation), nelems_(nelems)
      {
      }

      Array & operator=(const Array& a) 
      {
        nelems_ = a.nelems();
        storageLocationType_ = a.location();
        allocate();
        copy(a.data(), a.nelems());
        memcpy(data_, a.data(), sizeof(T)*nelems_);
      }

      void copy(T* dst, size_t nelems)
      {
        assert(nelems == nelems_);

        if (location() == Device) {
          assert(false);
        } else {
          memcpy(data_, dst, sizeof(T)*nelems);
        }
      }

      T* data() const {return data_;}
      inline size_t nelems() const {return nelems_;}
      inline StorageLocationType location() const {return storageLocationType_;}
      T& operator[](size_t idx) 
      {
        assert(idx < nelems());

        return data_[idx];
      }

      void allocate()
      {
        if (location() == Device) {
          assert(false); //TODO
        } else {
          data_ = new T[nelems()];
        }
      }

      void free()
      {
        if (location() == Device) {
          assert(false); //TODO
        } else {
          delete[] data_;
        }
      }

      ~Array()
      {
        free();
      }
  };
}

#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
  printf("Error at %s:%d\n",__FILE__,__LINE__);\
  abort();}} while(0)

namespace GPUUtils {
  typedef uint32_t ShMemEdgePos_t;

  enum SourceVertexExec_t
  {
    BlockLevel,
    DeviceLevel
  };

  const uint FULL_MASK = 0xffffffff;

  void printCudaMemInfo() 
  {
    size_t free = 0, total = 0;
    CHK_CU(cudaMemGetInfo(&free, &total));
    printf("free memory %ld\n", free);
  }

  __device__ inline int get_warp_mask_and_participating_threads (int condition, int& participating_threads, int& first_active_thread)
  {
    uint warp_mask = __ballot_sync(FULL_MASK, condition);
    first_active_thread = -1;
    participating_threads = 0;
    int qq = 0;
    while (qq < 32) {
      if ((warp_mask & (1U << qq)) == (1U << qq)) {
        if (first_active_thread == -1) {
          first_active_thread = qq;
        }
        participating_threads++;
      }
      qq++;
    }

    return warp_mask;
  }
  
  BCGPartition copyPartitionToGPU(BCG* bcg, GPUBCGPartition& gpuBCGPartition) {
    //TODO: Store gpuBCGPartition in Constant Memory

    auto size_offset = sizeof(Offset_t) * ((bcg->n_nodes) + 1);    
    CHK_CU(cudaMalloc(&gpuBCGPartition.d_offset, size_offset));
    Graph_t *d_offset_data;
    CHK_CU(cudaMalloc(&d_offset_data, sizeof(Graph_t) * (bcg->offset.size())));
    CHK_CU(cudaMemcpy(d_offset_data, &(bcg->offset[0]), sizeof(Graph_t) * (bcg->offset).size(), cudaMemcpyHostToDevice));

    decode_offset<<<min(8192L, bcg->n_nodes/256L), 256>>>(d_offset_data, gpuBCGPartition.d_offset, bcg->n_nodes, bcg->ubl);
    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, gpuBCGPartition.d_offset, gpuBCGPartition.d_offset, (bcg->n_nodes) + 1);
    CHK_CU(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, gpuBCGPartition.d_offset, gpuBCGPartition.d_offset, (bcg->n_nodes) + 1);
    
    auto size_graph = sizeof(Graph_t) * (bcg->graph).size();
    CHK_CU(cudaMalloc(&gpuBCGPartition.d_graph, size_graph));
    CHK_CU(cudaMemcpy(gpuBCGPartition.d_graph, &(bcg->graph[0]), size_graph, cudaMemcpyHostToDevice));

    BCGPartition device_bcg_partition = BCGPartition(gpuBCGPartition.d_graph, gpuBCGPartition.d_offset, bcg->n_nodes);
    
    printf("Copy %ld Bytes to GPU. (%ld graph, %ld offset)\n", size_graph + sizeof(Graph_t) * (bcg->offset).size(), size_graph, sizeof(Graph_t) * (bcg->offset).size());
    
    return device_bcg_partition;
  }
  
  __host__ __device__ int num_edges_to_warp_size (const EdgePos_t n_edges, SourceVertexExec_t src_vertex_exec) 
  {
    //Different warp sizes gives different performance. 32 is worst. adapative is a litter better.
    //Best is 4.
  #ifdef RANDOM_WALK
    return 1;
  #else
    if (src_vertex_exec == SourceVertexExec_t::BlockLevel) {
      //TODO: setting this to 4,8,or 16 gives error.
      if (n_edges < 4) 
        return 2;
      if (n_edges < 8)
        return 4;
      if (n_edges < 16)
        return 8;
      if (n_edges < 32)
        return 16;
      else
        return 32;
    } else {
      return 32;
    }
  #endif
  }

  float* gen_rand_on_gpu(size_t n_rands)
  {
    float* device_rand;
    cudaMalloc(&device_rand, n_rands*sizeof(float));
    curandGenerator_t gen;
    CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, clock()));
    CURAND_CALL(curandGenerateUniform(gen, device_rand, n_rands));

    return device_rand;
  }


  template<typename T>
  void printDeviceArray(T* array, int nelems, char sep)
  {
    T* tmp = new T[nelems];

    CHK_CU(cudaMemcpy(tmp, array, nelems*sizeof(T), cudaMemcpyDeviceToHost));
    for (int i = 0; i < nelems; i++) {
      if (tmp[i] == 4549983 || tmp[i] == 4017644)
      std::cout << i << ":" << tmp[i] << sep;
    }

    std::cout << std::endl;

    delete tmp;
  }

  template<typename T1, typename T2>
  void printKeyValuePairs(T1* keys, T2* values, int nelems, char sep)
  {
    for (int i = 0; i < nelems; i++) {
      if (values[i] == 4439958 || keys[i] == 4549983)
        std::cout << i << ": [" << keys[i] << ", " << values[i] << "]" << sep;
    }

    std::cout << std::endl;
  }

  template<typename T1, typename T2>
  void printDeviceKeyValuePairs(T1* keys, T2* values, int nelems, char sep)
  {
    T1* tmp1 = new T1[nelems];
    T2* tmp2 = new T2[nelems];

    CHK_CU(cudaMemcpy(tmp1, keys, nelems*sizeof(T1), cudaMemcpyDeviceToHost));
    CHK_CU(cudaMemcpy(tmp2, values, nelems*sizeof(T2), cudaMemcpyDeviceToHost));
    for (int i = 0; i < nelems; i++) {
      if (tmp1[i] == 4549983 || tmp1[i] == 4017644)
        std::cout << i << ": [" << tmp1[i] << ", " << tmp2[i] << "]" << sep;
    }

    std::cout << std::endl;
    delete tmp1;
    delete tmp2;
  }

  template<typename T>
  T* copyDeviceMemToHostMem(T* devPtr, size_t nelems)
  {
    T* hPtr = new T[nelems];

    CHK_CU(cudaMemcpy(hPtr, devPtr, nelems*sizeof(T), cudaMemcpyDeviceToHost));

    return hPtr;
  }
}

namespace LoadBalancing {
  enum LoadBalancingThreshold{
    GridLevel = 256,
    BlockLevel = 32,
    SubWarpLevel = 8,
    IdentityKernel = 0
  };

  enum LoadBalancingTBSizes {
    GridLevelTBSize = 512,
    BlockLevelTBSize = 512,
    SubWarpLevelTBSize = 32,
  };

  const bool EnableLoadBalancing = false;

  bool is_grid_level_assignment(const EdgePos_t num_roots) 
  {
    return num_roots >= LoadBalancingThreshold::GridLevel;
  }

  bool is_block_level_assignment(const EdgePos_t num_roots) 
  {
    return num_roots < LoadBalancingThreshold::GridLevel and num_roots >= LoadBalancingThreshold::BlockLevel;
  }

  bool is_subwarp_level_assignment(const EdgePos_t num_roots) 
  {
    return num_roots < LoadBalancingThreshold::SubWarpLevel;
  }

};

#endif