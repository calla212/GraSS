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
#define WARP_LEN 32
#define BLOCK_DIM 256

#define MB32 1024 * 1024 * 32

struct BCG {
    std::vector<Graph_t> graph;
    std::vector<Graph_t> offset;
    VertexID_t n_nodes;
    uint8_t ubl;

    BCG() {
        this->graph.clear();
        this->offset.clear();
        this->n_nodes = 0;
        ubl = 0;
        return;
    }
};

struct BCGVertex {
    VertexID_t vertex;
    VertexID_t outd;
    VertexID_t block_len, block_num, om_num, first_pos;
    int head_bit, body_bit;
    Offset_t offset;
    const Graph_t* graph;

    #define USE_SM
    #ifdef USE_SM
        int CACHE_SIZE = 0;
        Graph_t* sm_graph;
    
        __device__
        VertexID_t get_sm_num(int _CACHE_SIZE, Graph_t* _sm_graph) {
            CACHE_SIZE = _CACHE_SIZE;
            sm_graph = _sm_graph;
            auto read_num = offset + head_bit * block_num + body_bit * (outd - block_num);
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
    #else
        __device__
        Graph_t get_32b(Offset_t bit_offset) {
            Offset_t chunk = bit_offset / GRAPH_LEN;
            Graph_t h_bit = graph[chunk];
            Graph_t l_bit = graph[chunk + 1];
            Offset_t chunk_offset = bit_offset % GRAPH_LEN;
            return __funnelshift_l(l_bit, h_bit, chunk_offset);
        }
    #endif

    __device__
    int get_vlc_bit(Offset_t &bit_offset) {
        auto tmp = get_32b(bit_offset);
        auto x = __clz(tmp);
        bit_offset += x;
        return x + 1;
    }

    __device__
    Graph_t decode(Offset_t& bit_offset, int vlc_bit) {
        auto ret = get_32b(bit_offset) >> (32 - vlc_bit);
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
    BCGVertex(VertexID_t _vertex, const Graph_t* _graph, const Offset_t _offset) : vertex(_vertex), graph(_graph), offset(_offset) {
        // printf("-----%d %p %ld\n", vertex, graph, offset);
        
        outd = decode_vlc(offset);
        head_bit = decode_vlc(offset);
        body_bit = decode_vlc(offset);
        first_pos = decode_vlc(offset);
        // printf("%d : %d %d %d\n", vertex, outd, head_bit, body_bit);
        // if (vertex == 9) printf("%d : %d %d %d\n", vertex, outd, head_bit, body_bit);

        VertexID_t nsqrt = sqrt((float)outd);
        if (nsqrt * nsqrt == outd) {
            block_num = block_len = nsqrt;
            om_num = 0;
        }
        else {
            block_num = nsqrt + 1;
            block_len = outd / block_num;
            om_num = outd - block_num * block_len;
        }

        Offset_t chunk = offset / GRAPH_LEN;
        graph += chunk;
        offset -= chunk * GRAPH_LEN;

        return;
    }

    __device__
    VertexID_t decode_head(VertexID_t neighbor_block, Offset_t& bit_offset) {
        VertexID_t ret = decode(bit_offset, head_bit);
        // printf("   Trn=%d _RET=%d\n NB=%d--FP=%d\n", vertex, ret, neighbor_block, first_pos);
        if (neighbor_block >= first_pos) ret = vertex + ret + 1;
        else ret = vertex - ret - 1;
        return ret;
    }

    __device__
    void decode_body(VertexID_t& cur_vertex, Offset_t& bit_offset) {
        cur_vertex += decode(bit_offset, body_bit) + 1;
        return;
    }

    __device__
    VertexID_t get_vertex(VertexID_t neighbor) {
        bool flag = neighbor < (block_len + 1) * om_num;
        neighbor -= (!flag) * (block_len + 1) * om_num;
        VertexID_t neighbor_block = neighbor / (block_len + flag);
        auto neighbor_id = neighbor - neighbor_block * (block_len + flag);
        neighbor_block += (!flag) * om_num;

        auto bit_offset = offset + neighbor_block * (body_bit * block_len - body_bit + head_bit) + min(neighbor_block, om_num) * body_bit;

        VertexID_t cur_vertex = decode_head(neighbor_block, bit_offset);

        while (neighbor_id--) decode_body(cur_vertex, bit_offset);

        return cur_vertex;
    }
};

struct GlobalDecoder {
    VertexID_t vertex;
    VertexID_t outd;
    VertexID_t block_len, block_num, first_pos;
    VertexID_t om_num, sm_len, sm_om_num;
    int head_bit, body_bit;
    Offset_t offset, body_offset;
    const Graph_t* graph;

    int CACHE_SIZE = 0;
    VertexID_t* sm_graph;

    __device__
    void set_sm_num(int _CACHE_SIZE, VertexID_t* _sm_graph) {
        CACHE_SIZE = _CACHE_SIZE;
        sm_graph = _sm_graph;
        sm_len = CACHE_SIZE / block_num;
        sm_om_num = CACHE_SIZE - sm_len * block_num;
        return;
    }

    __device__
    VertexID_t get_len(VertexID_t neighbor_block) {
        return min(sm_len + (neighbor_block < sm_om_num), block_len + (neighbor_block < om_num));
    }

    __device__
    Graph_t get_32b(Offset_t bit_offset) {
        Offset_t chunk = bit_offset / GRAPH_LEN;
        Graph_t h_bit = graph[chunk];
        Graph_t l_bit = graph[chunk + 1];
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

    __device__ inline
    Graph_t decode(Offset_t bit_offset, int fix_bit) {
        return get_32b(bit_offset) >> (32 - fix_bit);
    }

    __device__
    VertexID_t decode_vlc(Offset_t &bit_offset) {
        auto vlc_bit = get_vlc_bit(bit_offset);
        auto ret = decode(bit_offset, vlc_bit) - 1;
        bit_offset += vlc_bit;
        return ret;
    }

    __device__
    GlobalDecoder(VertexID_t _vertex, const Graph_t* _graph, const Offset_t _offset) : vertex(_vertex), graph(_graph), offset(_offset) {
        // printf("-----%d %p %ld\n", vertex, graph, offset);
        
        outd = decode_vlc(offset);
        head_bit = decode_vlc(offset);
        body_bit = decode_vlc(offset);
        first_pos = decode_vlc(offset);
        // printf("%d : %d %d %d\n", vertex, outd, head_bit, body_bit);

        VertexID_t nsqrt = sqrt((float)outd);
        if (nsqrt * nsqrt == outd) {
            block_num = block_len = nsqrt;
            om_num = 0;
        }
        else {
            block_num = nsqrt + 1;
            block_len = outd / block_num;
            om_num = outd - block_num * block_len;
        }

        Offset_t chunk = offset / GRAPH_LEN;
        graph += chunk;
        offset -= chunk * GRAPH_LEN;
        return;
    }

    __device__
    VertexID_t get_head(VertexID_t neighbor_block) {
        body_offset = offset + neighbor_block * (body_bit * block_len - body_bit + head_bit) + min(neighbor_block, om_num) * body_bit;
        VertexID_t ret = decode(body_offset, head_bit);
        body_offset += head_bit;
        // auto _ret = ret;
        // printf("   Trn=%d _RET=%d\n NB=%d--FP=%d\n", vertex, ret, neighbor_block, first_pos);
        if (neighbor_block >= first_pos) ret = vertex + ret + 1;
        else ret = vertex - ret - 1;
        // if (ret < 0) printf("%d %d %d==%d(%d)\n", vertex, outd, ret, _ret, head_bit);
        // assert(ret >= 0);
        // assert(ret <= 4846609);
        return ret;
    }

    __device__
    VertexID_t get_body(VertexID_t neighbor_id) {
        // printf("Trn=%d HeadInfo OD=%d=%d*%d Hb=%d Bb=%d FP=%d\n", vertex, outd, block_num, block_len, head_bit, body_bit, first_pos);
        // printf("Trn=%d SpInfo NB=%d=%d*%d+%d BO=%lld\n", vertex, neighbor, neighbor_block, block_len, neighbor_id, bit_offset);
        // printf("Trn=%d NB=%d HV=%d BO=%ld\n", vertex, neighbor, cur_vertex, bit_offset);
        return decode(body_offset + neighbor_id * body_bit, body_bit) + 1;
    }

    __device__
    VertexID_t get_vertex(VertexID_t neighbor) {
        bool flag = neighbor < (block_len + 1) * om_num;
        neighbor -= (!flag) * (block_len + 1) * om_num;
        VertexID_t neighbor_block = neighbor / (block_len + flag);
        auto neighbor_id = neighbor - neighbor_block * (block_len + flag);
        neighbor_block += (!flag) * om_num;

        flag = neighbor_block < sm_om_num;
        VertexID_t ret;
        
        if (neighbor_block >= CACHE_SIZE) ret = get_head(neighbor_block);
        else {
            body_offset = offset + neighbor_block * (body_bit * block_len - body_bit + head_bit) + min(neighbor_block, om_num) * body_bit + head_bit;
            ret = sm_graph[neighbor_block * sm_len + min(neighbor_block, sm_om_num) + min(neighbor_id, (sm_len + flag - 1))];
        }

        for (int this_id = sm_len + flag; this_id <= neighbor_id; ++this_id) {
            ret += decode(body_offset + this_id * body_bit - body_bit, body_bit) + 1;
        }
        return ret;
    }

    __device__
    VertexID_t get_vertex_debug(VertexID_t neighbor) {
        auto _neighbor = neighbor;
        bool flag = neighbor < (block_len + 1) * om_num;
        neighbor -= (!flag) * (block_len + 1) * om_num;
        VertexID_t neighbor_block = neighbor / (block_len + flag);
        auto neighbor_id = neighbor - neighbor_block * (block_len + flag);
        neighbor_block += (!flag) * om_num;

        flag = neighbor_block < sm_om_num;
        VertexID_t ret;
        
        if (neighbor_block >= CACHE_SIZE) ret = get_head(neighbor_block);
        else {
            body_offset = offset + neighbor_block * (body_bit * block_len - body_bit + head_bit) + min(neighbor_block, om_num) * body_bit + head_bit;
            ret = sm_graph[neighbor_block * sm_len + min(neighbor_block, sm_om_num) + min(neighbor_id, (sm_len + flag - 1))];
        }

        printf("DEBUG Trn=%d AT=%d(%d,%d) %d[%d]\n", vertex, _neighbor, neighbor_block, neighbor_id, ret, neighbor_block * sm_len + min(neighbor_block, sm_om_num) + min(neighbor_id, (sm_len + flag - 1)));

        for (int this_id = sm_len + flag; this_id <= neighbor_id; ++this_id) {
            printf("a(%d) %d %d\n", vertex, this_id, ret);
            ret += decode(body_offset + this_id * body_bit - body_bit, body_bit) + 1;
            printf("b(%d) %d %d\n", vertex, this_id, ret);
        }
        return ret;
    }

    __device__
    void write_vertex(VertexID_t v, VertexID_t neighbor_block, VertexID_t neighbor_id) {
        
        auto pos = neighbor_block * sm_len + (neighbor_block < sm_om_num ? neighbor_block : sm_om_num) + neighbor_id;
        // printf("%d=%d*%d+(%d %d)+%d\n", pos, neighbor_block, sm_len, neighbor_block, sm_om_num, neighbor_id);
        if (pos < CACHE_SIZE) {
            // assert(v >= 0);
            // if (v > 4846609) {
            //     printf("%d (%d %d) %d\n", vertex, neighbor_block, neighbor_id, v);
            // }
            // assert(v <= 4846609);
            // assert(pos >= 0);
            // assert(sm_graph != nullptr);
            sm_graph[pos] = v;
        }
        return;
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
