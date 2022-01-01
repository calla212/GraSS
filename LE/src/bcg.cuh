#ifndef __BCG_HPP__
#define __BCG_HPP__

#include <vector>
#include <bitset>

typedef uint32_t Graph_t;
typedef uint64_t Offset_t;
typedef int32_t VertexID_t;
typedef int32_t VertexID;
typedef int32_t EdgePos_t;

#define GRAPH_LEN 32
#define SM_SIZE 8

struct BCG {
    std::vector<Graph_t> graph, offset;
    VertexID_t n_nodes;
    uint8_t ubl;

    BCG() {
        this->graph.clear();
        this->offset.clear();
        this->n_nodes = 0;
        ubl = 0;
        return;
    }

    // void print(void) {
    //     std::cout << "Graph has " << n_nodes << "vertices.\n";
    //     for (int i = 0; i < offset.size(); ++i) std::cout << "Offset_" << i << " : " << offset[i] << std::endl;
    //     for (auto i : graph) std::cout << std::bitset<sizeof(Graph_t)*8>(i) << " ";
    //     std::cout << "\n-----------------------------------------\n";
    //     return;
    // }
};

struct BCGVertex {
    VertexID_t vertex;
    VertexID_t outd;
    VertexID_t chunk_num, chunk_len, om_num, first_pos;
    int rock_bit, sea_bit;
    Offset_t offset;
    const Graph_t* graph;
    bool len64;

    int CACHE_SIZE = 0;
    Graph_t* sm_graph;

    __device__
    VertexID_t get_sm_num(int _CACHE_SIZE, Graph_t* _sm_graph) {
        CACHE_SIZE = _CACHE_SIZE;
        sm_graph = _sm_graph;
        auto read_num = offset + (chunk_num + 1) * rock_bit + (outd - chunk_num - 1) * (sea_bit + 1);
        read_num = (read_num + GRAPH_LEN) / GRAPH_LEN;
        return read_num < CACHE_SIZE ? read_num : CACHE_SIZE;
    }

    __device__ __forceinline__
    Graph_t get_item(Offset_t g_offset) {
        if (g_offset < CACHE_SIZE) return sm_graph[g_offset];
        return graph[g_offset];
    }

    __device__
    Graph_t get_32b(Offset_t bit_offset) {
        Offset_t chunk = bit_offset / GRAPH_LEN;
        Graph_t h_bit = get_item(chunk);
        Graph_t l_bit = get_item(chunk + 1);
        Offset_t chunk_offset = bit_offset % GRAPH_LEN;
        return __funnelshift_l(l_bit, h_bit, chunk_offset);
    }
    __device__
    int get_vlc_bit(Offset_t &bit_offset) {
        auto tmp = get_32b(bit_offset);
        auto x = __clz(tmp);
        bit_offset += x;
        return x + 1;
    }

    __device__
    Graph_t decode(Offset_t& bit_offset, int vlc_bit) {
        auto ret = get_32b(bit_offset) >> (GRAPH_LEN - vlc_bit);
        bit_offset += vlc_bit;
        return ret;
    }

    __device__
    VertexID_t decode_vlc(Offset_t &bit_offset) {
#ifdef ZETA
        auto vlc_bit = get_vlc_bit(bit_offset);
        ++bit_offset;
        return decode(bit_offset, vlc_bit * ZETA) - 1;
#else
        auto vlc_bit = get_vlc_bit(bit_offset);
        return decode(bit_offset, vlc_bit) - 1;
#endif
    }

    __device__
    VertexID_t decode_vlc_signed(Offset_t &bit_offset) {
#ifdef ZETA
        auto vlc_bit = get_vlc_bit(bit_offset);
        ++bit_offset;
        return decode(bit_offset, vlc_bit * ZETA) - 1;
#else
    
        Offset_t type_id = bit_offset / GRAPH_LEN;
        Graph_t offset_id = bit_offset - type_id * GRAPH_LEN;
        Graph_t data[2];
        data[0] = get_item(type_id);
        data[1] = get_item(type_id + 1);
        auto vlc_bit = __clz(__funnelshift_l(data[1], data[0], offset_id));

        offset_id += vlc_bit++;
        bool flag = offset_id >= GRAPH_LEN;
        offset_id -= flag * GRAPH_LEN;
        if (flag && offset_id + vlc_bit + 1 > GRAPH_LEN) data[0] = get_item(type_id + 2);
        VertexID_t ret = (__funnelshift_l(data[!flag], data[flag], offset_id) >> (GRAPH_LEN - vlc_bit)) - 1;

        offset_id += vlc_bit;
        flag = flag ^ (offset_id >= GRAPH_LEN);
        offset_id -= GRAPH_LEN * (offset_id >= GRAPH_LEN);
        bit_offset += (vlc_bit << 1);
        if ((data[flag] >> GRAPH_LEN - offset_id - 1) & 1) return -ret;
        return ret;
#endif
    }

    __device__
    BCGVertex(VertexID_t _vertex, const Graph_t* _graph, const Offset_t _offset) : vertex(_vertex), graph(_graph), offset(_offset) {
        outd = decode_vlc(offset);
        if (outd == 0) return;
        // printf("%d : %ld\n", vertex, offset);
        rock_bit = decode_vlc(offset);
        if (outd == 1) return;
        sea_bit = decode_vlc(offset);
        first_pos = decode_vlc(offset);
        // printf("%d (%d) : %d %d %d\n", vertex, mv_vertex, outd, rock_bit, sea_bit);

        auto _outd = outd - 1;
        VertexID_t nsqrt = sqrt((float)_outd);
        om_num = _outd - nsqrt * nsqrt;
        chunk_num = nsqrt + (om_num > nsqrt);
        if (chunk_num) chunk_len = _outd / chunk_num;
        om_num = _outd - chunk_num * chunk_len;

        Offset_t type_id = offset / GRAPH_LEN;
        graph += type_id;
        offset -= type_id * GRAPH_LEN;
        return;
    }

    // __device__
    // VertexID_t decode_head(VertexID_t neighbor_chunk, Offset_t& bit_offset) {
    //     bit_offset += block_bit_num * neighbor_chunk;
    //     VertexID_t ret = decode(bit_offset, head_bit);
    //     // printf("   Trn=%d _RET=%d\n NB=%d--FP=%d\n", vertex, ret, neighbor_chunk, first_pos);
    //     if (neighbor_chunk >= first_pos) ret = vertex + ret + 1;
    //     else ret = vertex - ret - 1;
    //     return ret;
    // }

   
    // __device__
    // void decode_body(VertexID_t& cur_vertex, Offset_t& bit_offset) {
    //     cur_vertex += decode(bit_offset, body_bit) + 1;
    //     return;
    // }

    __device__
    VertexID_t decode_sea(Offset_t bit_offset) {
        Offset_t type_id = bit_offset / GRAPH_LEN;
        Graph_t offset_id = bit_offset - type_id * GRAPH_LEN;
        Graph_t data[2];
        data[0] = get_item(type_id);
        data[1] = 0;
        if (offset_id + sea_bit + 1 > GRAPH_LEN) data[1] = get_item(type_id + 1);
        VertexID_t ret = __funnelshift_l(data[1], data[0], offset_id) >> (GRAPH_LEN - sea_bit);
        offset_id += sea_bit;
        bool flag = offset_id >= GRAPH_LEN;
        offset_id -= flag * GRAPH_LEN;
        // printf("%d %d\n", vertex, data[flag] >> (GRAPH_LEN - offset_id - 1));
        if ((data[flag] >> (GRAPH_LEN - offset_id - 1)) & 1) return 0 - ret;
        return ret;
    }

    __device__
    Graph_t _decode(Offset_t& bit_offset, int vlc_bit) {
        auto ret = get_32b(bit_offset) >> (32 - vlc_bit);
        bit_offset += vlc_bit;
        return ret;
    }

    __device__
    VertexID_t get_vertex(VertexID_t neighbor) {
        if (outd == 0) return -1;
        if (outd == 1) return rock_bit;
        
        VertexID_t neighbor_chunk = 0, this_len = chunk_len + 1;
        bool flag = neighbor >= (om_num * this_len);
        neighbor -= flag * om_num * this_len;
        this_len -= flag;
        if (this_len) neighbor_chunk = neighbor / this_len;
        auto neighbor_id = neighbor - neighbor_chunk * this_len;
        neighbor_chunk += flag * om_num;
        
        // Graph_t data[2];
        Offset_t decode_offset = (offset + neighbor_chunk * rock_bit);
        // auto _decode_offset = decode_offset;
        // Offset_t type_id = decode_offset / GRAPH_LEN;
        // short offset_id = decode_offset - type_id * GRAPH_LEN;
        
        // data[0] = get_item(type_id);
        // data[1] = get_item(type_id + 1);
        
        // VertexID_t r1 = __funnelshift_l(data[1], data[0], offset_id) >> (GRAPH_LEN - rock_bit);
        VertexID_t r1 = _decode(decode_offset, rock_bit);
        // assert(r1 == _r1);
        
        if (neighbor_chunk < first_pos) r1 = vertex - r1;
        else r1 = vertex + r1;

        if (neighbor_id == 0) return r1;        

        // offset_id += rock_bit;
        // flag = offset_id >= GRAPH_LEN;
        // offset_id -= flag * GRAPH_LEN;
        // if (flag) data[!flag] = get_item(type_id + 2);
        // VertexID_t r2 = __funnelshift_l(data[!flag], data[flag], offset_id) >> (GRAPH_LEN - rock_bit);
        VertexID_t r2 = _decode(decode_offset, rock_bit);
        // assert(r2 == _r2);

        if (neighbor_chunk + 1 < first_pos) r2 = vertex - r2;
        else r2 = vertex + r2;
      
        decode_offset = offset + chunk_num * rock_bit + rock_bit + ((chunk_len - 1) * neighbor_chunk + min(om_num, neighbor_chunk) + neighbor_id - 1) * (sea_bit + 1);
        // _decode_offset = decode_offset;
        // type_id = decode_offset / GRAPH_LEN;
        // offset_id = decode_offset - type_id * GRAPH_LEN;
        // data[0] = get_item(type_id);
        // data[1] = get_item(type_id + 1);
        // VertexID_t sea = __funnelshift_l(data[1], data[0], offset_id) >> (GRAPH_LEN - sea_bit);
        VertexID_t sea;
        if (sea_bit < 32) {
            sea = _decode(decode_offset, sea_bit + 1);
            flag = sea & 1;
            sea >>= 1;
        }
        else {
            sea = _decode(decode_offset, sea_bit);
            flag = _decode(decode_offset, 1);
        }
        // assert(sea == _sea);
        // offset_id += sea_bit;
        // flag = offset_id >= GRAPH_LEN;
        // offset_id -= flag * GRAPH_LEN;
        // flag = (data[flag] >> (GRAPH_LEN - offset_id - 1)) & 1;
        
        return (VertexID_t)((r2 - r1) / this_len) * neighbor_id + r1 + (flag ? -1 : 1) * sea;
    }
};

struct BCGPartition {
    ~BCGPartition() {}
//   const VertexID first_vertex_id;
//   const VertexID last_vertex_id;
    const Graph_t *graph;
    const Offset_t *offset;
    const VertexID_t n_nodes;

     __host__
    BCGPartition (const Graph_t *_graph, const Offset_t *_offset, const VertexID_t _n_nodes) : 
                                graph(_graph), offset(_offset), n_nodes(_n_nodes) {}
};

struct GPUBCGPartition {
    BCGPartition* d_bcg;
    Graph_t* d_graph;
    Offset_t* d_offset;
};


__global__
void decode_offset(Graph_t *offset_data, Offset_t *offset, VertexID_t n_nodes, uint8_t ubl) {
    VertexID_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    // __shared__ uint32_t sm_cache[];
    for (VertexID_t gos = 0; gos < n_nodes; gos += blockDim.x * gridDim.x) {
        // Offset_t sm_os = gos * ubl / GRAPH_LEN;
        // sm_cache[tid] = offset[gos + tid];
        // __syncthreads();

        VertexID_t u = gos + tid;
        if (u < n_nodes) {
            Offset_t tmp = 1ll * u * ubl;
            // if (u < 8) printf("%d %ld\n", u, tmp);
            // auto type_id = tmp / GRAPH_LEN - sm_os;
            Offset_t type_id = tmp / GRAPH_LEN;
            int16_t bit_offset = tmp - type_id * GRAPH_LEN, bit_len = ubl;

            // assert(type_id >= 0);
            // if (type_id >= 1920276) {
            //     printf("%ld %d %ld\n", tmp, GRAPH_LEN, type_id);
            // }
            // assert(type_id < 1920276);
            // tmp = (n_nodes << bit_offset) >> max(GRAPH_LEN - bit_len, bit_offset);
            tmp = (offset_data[type_id] << bit_offset) >> max(GRAPH_LEN - bit_len, bit_offset);
            bit_len -= (GRAPH_LEN - bit_offset);
            // if (u < 8) printf("   %d %d\n", tid, bit_len);


            if (bit_len > 0) {
                tmp = (tmp << min(bit_len, 32)) | (offset_data[type_id + 1] >> max(GRAPH_LEN - bit_len, 0));
                bit_len -= GRAPH_LEN;
                if (bit_len > 0) {
                    tmp = (tmp << min(bit_len, 32)) | (offset_data[type_id + 2] >> max(GRAPH_LEN - bit_len, 0));
                }
            }

            // assert(u >= 0);
            // assert(u < 3072441);
            // if (u < 8) printf("      %d %d %lld\n", tid, u, tmp);
            // if (u >= 384054 && u <= 384058) printf("      %d %d %lld\n", tid, u, tmp);
            offset[u] = tmp;
        }
    }

    return;
}


#endif
