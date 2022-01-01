#ifndef CGR_COMPRESSOR_HPP
#define CGR_COMPRESSOR_HPP

#include <string>
#include <iostream>
#include <cassert>
#include <algorithm>
#include <vector>
#include <deque>
#include <cmath>


using size_type = int64_t;
using bits = std::vector<bool>;

int get_bit_num(int64_t num) {
    if (num < 0) num = 0 - num;
    if (num == 0) return 1;
    int ret = 0;
    while (num > 0) {num >>= 1; ++ret;}
    return ret;
}

class Compressor {

    const size_type PRE_ENCODE_NUM = 16 * 1024 * 1024;

    int zeta_k;

    size_type num_node;
    size_type num_edge;
    int max_head_bit, max_body_bit;
    std::vector<std::pair<size_type, size_type>> edges;
    std::vector<std::vector<size_type>> adjlist;

    class NodeCompress {
    public:
        size_type node;
        size_type outd;
        size_type chunk_num, chunk_len, om_num, first_pos;
        int rock_bit, sea_bit;

        std::vector<size_type> rock_node;
        std::vector<size_type> sea_node;

        bits bit_arr;
        std::deque<bool> len_arr;

        void set_chunks(const std::vector<size_type>& neighbors) {
            if (outd == 0) {
                chunk_len = chunk_num = rock_bit = 0;
                // std::cout << node << " 0000000" << std::endl;
                return;
            }

            if (outd == 1) {
                rock_bit = neighbors[0];
                return;
            }
            
            auto _outd = outd - 1;
            size_type nsqrt = sqrt(_outd);
            om_num = _outd - nsqrt * nsqrt;
            chunk_num = nsqrt + (om_num > nsqrt);
            if (chunk_num) chunk_len = _outd / chunk_num;
            om_num = _outd - chunk_num * chunk_len;
            --chunk_len;
            
            rock_bit = get_bit_num(std::max(abs(neighbors[0] - node), abs(neighbors[outd - 1] - node)));
            return;
        }

        void init(size_type _node, const std::vector<size_type>& neighbors) {
            node = _node;
            outd = neighbors.size();
            sea_bit = 0;
            first_pos = -1;
            set_chunks(neighbors);
            rock_node.clear();
            sea_node.clear();
            bit_arr.clear();
            return;
        }
    };

    std::vector<NodeCompress> cnodes;
    std::vector<bits> gamma_code;
    std::vector<bits> zeta_code;

public:
    explicit Compressor(int _zeta_k = 3) : zeta_k(_zeta_k), num_node(0), num_edge(0) {}

    bool load_graph(const std::string &file_path) {
        FILE *f = fopen(file_path.c_str(), "r");

        if (f == 0) {
            std::cout << "file cannot open!" << std::endl;
            abort();
        }

        size_type u = 0, v = 0;
        this->num_node = 0;
        while (fscanf(f, "%ld %ld", &u, &v) > 0) {
            assert(u >= 0);
            assert(v >= 0);
            this->num_node = std::max(this->num_node, u + 1);
            this->num_node = std::max(this->num_node, v + 1);
            if (!this->edges.empty()) {
                assert(this->edges.back().first <= u);
                if (this->edges.back().first == u) {
                    if (this->edges.back().second == v) // printf("%d %d %d\n", u, v, this->edges.back().second);
                        continue;
                    assert(this->edges.back().second <= v);
                }
            }
            this->edges.emplace_back(std::pair<size_type, size_type>(u, v));
        }

        this->num_edge = this->edges.size();

        this->adjlist.resize(this->num_node);
        for (auto edge : this->edges) {
            this->adjlist[edge.first].emplace_back(edge.second);
        }
        fclose(f);

        return true;
    }

    bool write_cgr(const std::string &dir_path) {
        bits graph;

        FILE *of_graph = fopen((dir_path + ".graph").c_str(), "w");

        if (of_graph == 0) {
            std::cout << "graph file cannot create!" << std::endl;
            abort();
        }

        this->write_data(of_graph);
        fclose(of_graph);
    
        FILE *of_offset = fopen((dir_path + ".offset").c_str(), "w");

        if (of_offset == 0) {
            std::cout << "offset file cannot create!" << std::endl;
            abort();
        }

        this->write_offset3(of_offset);
        fclose(of_offset);

        return true;
    }

    void write_data(FILE* &of) {
        std::vector<unsigned char> buf;

        unsigned char cur = 0;
        int bit_count = 0;

        for (size_type i = 0; i < this->num_node; i++) {
            for (auto bit : this->cnodes[i].bit_arr) {
                cur <<= 1;
                if (bit) cur++;
                bit_count++;
                if (bit_count == 8) {
                    buf.emplace_back(cur);
                    cur = 0;
                    bit_count = 0;
                }
            }
        }

        // for (size_type i = 0; i < this->num_node; i++) {
        //     std::cout << i << std::endl;
        //     for (auto bit : this->cnodes[i].bit_arr) std::cout << bit;
        //     std::cout << std::endl;
        // }

        if (bit_count) {
            while (bit_count < 8) cur <<= 1, bit_count++;
            buf.emplace_back(cur);
        }

        fwrite(buf.data(), sizeof(unsigned char), buf.size(), of);
    }

    void write_offset1(FILE* &of) {
        // binary

        size_type last_offset = 0;
        for (size_type i = 0; i < this->num_node; i++) last_offset += this->cnodes[i].bit_arr.size();

        bool kind = last_offset > (1ll << 31);
        // std::cout << last_offset << " " << kind << std::endl;
        fwrite(&kind, sizeof(bool), 1, of);
        
        last_offset = 0;
        if (kind) {
            for (size_type i = 0; i < this->num_node; i++) {
                // if (i <= 8) std::cout << last_offset << std::endl;
                // if (i >= 384054 && i <= 384058) std::cout << i << ' ' << last_offset << std::endl;
                last_offset += this->cnodes[i].bit_arr.size();
                fwrite(&last_offset, sizeof(int64_t), 1, of);
            }
        }
        else {
            for (size_type i = 0; i < this->num_node; i++) {
                last_offset += this->cnodes[i].bit_arr.size();
                fwrite(&last_offset, sizeof(int32_t), 1, of);
            }
        }
        return;
    }

    void write_offset3(FILE* &of) {
        // binary + min_len + sum

        size_type last_offset = 0;

        std::vector<unsigned char> buf;
        unsigned char cur = 0;
        int bit_count = 0;
        int max_len = 0;

        for (size_type i = 0; i < this->num_node; i++) {
            if (this->cnodes[i].bit_arr.size() > max_len)
                max_len = this->cnodes[i].bit_arr.size();
        }
        // std::cout << max_len << ' ' << get_bit_num(max_len) << std::endl;
        max_len = get_bit_num(max_len);

        bits len_bit;
        append_bit(len_bit, max_len, 8);
        for (auto bit : len_bit) {
            cur <<= 1;
            if (bit) cur++;
            bit_count++;
            if (bit_count == 8) {
                buf.emplace_back(cur);
                cur = 0;
                bit_count = 0;
            }
        }
        
        for (size_type i = 0; i < this->num_node; i++) {
            auto len = this->cnodes[i].bit_arr.size();

            // if (i < 8) std::cout << len << std::endl;
            // if (i >= 384054 && i <= 384058) std::cout << len << std::endl;
            
            auto bit_cnt = get_bit_num(len);

            for (int i = bit_cnt; i < max_len; ++i) {
                cur <<= 1;
                bit_count++;
                if (bit_count == 8) {
                    buf.emplace_back(cur);
                    cur = 0;
                    bit_count = 0;
                }
            }

            for (auto bit : this->cnodes[i].len_arr) {
                cur <<= 1;
                if (bit) cur++;
                bit_count++;
                if (bit_count == 8) {
                    buf.emplace_back(cur);
                    cur = 0;
                    bit_count = 0;
                }
            }
        }

        if (bit_count) {
            while (bit_count < 8) cur <<= 1, bit_count++;
            buf.emplace_back(cur);
        }

        // std::cout << buf.size() << std::endl;

        fwrite(buf.data(), sizeof(unsigned char), buf.size(), of);
    }


    void encode_node_s1(const size_type node) {
        auto &cnode = this->cnodes[node];
        auto &neighbors = this->adjlist[node];

        cnode.init(node, neighbors);
        if (cnode.outd <= 1) return;
        
        size_type cur_id = 0;
        for (size_type chunk_id = 0; chunk_id < cnode.chunk_num; ++chunk_id) {
            auto rock_l = neighbors[cur_id];
            auto cur_len = cnode.chunk_len + (chunk_id < cnode.om_num);
            auto rock_r = neighbors[cur_id + cur_len + 1];
            auto cur_dis = (rock_r - rock_l) / (cur_len + 1);

            auto cur = neighbors[cur_id] - node;
            if (cur < 0) {
                cur = 0 - cur;
                cnode.first_pos = chunk_id;
            }
            cnode.rock_node.emplace_back(cur);

            ++cur_id;
            for (int i = 1; i <= cur_len && cur_id < cnode.outd; ++i, ++cur_id) {
                cur = neighbors[cur_id] - (rock_l + i * cur_dis);
                // std::cout << node << "[" << neighbors[cur_id] << "] " << chunk_id << "--" << cnode.block_num << " " << i << "==" << cnode.block_len << " " << cur_id <<" ->" << cur << std::endl;
                cnode.sea_node.emplace_back(cur);
                cnode.sea_bit = std::max(cnode.sea_bit, get_bit_num(cur));
            }
        }

        auto cur = neighbors[cur_id] - node;
        if (cur < 0) {
            cur = 0 - cur;
            cnode.first_pos = cnode.chunk_num;
        }
        cnode.rock_node.emplace_back(cur);
        ++cnode.first_pos;
        
        return;
    }
    
    void encode_node_s2(const size_type node) {

        // std::cout << node << std::endl;
        auto &cnode = this->cnodes[node];

        // std::cout << cnode.outd << " " << cnode.rock_bit << " " << cnode.sea_bit << " " << cnode.mv_dis << std::endl;
        // std::cout << cnode.chunk_num << "*" << cnode.chunk_len << "+" << cnode.om_num << std::endl;

// #define ZETA
#ifdef ZETA
        append_zeta(cnode.bit_arr, cnode.outd);
        if (max_body_bit == -1) {
            append_zeta(cnode.bit_arr, cnode.head_bit);
            append_zeta(cnode.bit_arr, cnode.body_bit);
        }
        append_zeta(cnode.bit_arr, cnode.first_pos);
#else
        append_gamma(cnode.bit_arr, cnode.outd);
        
        if (cnode.outd == 0) return;

        append_gamma(cnode.bit_arr, cnode.rock_bit);
        if (cnode.outd == 1) return;
        append_gamma(cnode.bit_arr, cnode.sea_bit);
        append_gamma(cnode.bit_arr, cnode.first_pos);
#endif

#define GATHER_HEAD
#ifdef GATHER_HEAD
        for (auto cur : cnode.rock_node) append_bit(cnode.bit_arr, cur, cnode.rock_bit);
        for (auto cur : cnode.sea_node) append_signed_bit(cnode.bit_arr, cur, cnode.sea_bit);
        // if (node == 740944) {
        //     for (auto cur : cnode.rock_node) std::cout << cur << ' ';
        //     std::cout << std::endl;
        //     for (auto cur : cnode.sea_node) std::cout << cur << ' ';
        //     std::cout << std::endl;
        // }
#else
        auto &neighbors = this->adjlist[node];
        size_type cur_id = 0;
        for (size_type chunk_id = 0; chunk_id < cnode.block_num; ++chunk_id) {
            auto cur = neighbors[cur_id] - node;
            if (cur < 0) {
                cnode.first_pos = chunk_id;
                cur = 0 - cur;
            }
            append_bit(cnode.bit_arr, cur - 1, this_head_bit);

            ++cur_id;
            for (int i = 1; i < cnode.block_len && cur_id < cnode.outd; ++i, ++cur_id) {
                cur = neighbors[cur_id] - neighbors[cur_id - 1] - 1;
                append_bit(cnode.bit_arr, cur, this_body_bit);
            }
        }

        for (; cur_id < cnode.outd; ++cur_id) {
            auto cur = neighbors[cur_id] - neighbors[cur_id - 1] - 1;
            append_bit(cnode.bit_arr, cur, this_body_bit);
        }
#endif
        return;
    }

    void encode_offset(const size_type node) {
        auto len = this->cnodes[node].bit_arr.size();
        auto bit_cnt = get_bit_num(len);
        
        size_type mask = 1ll << (bit_cnt - 1);
        while (mask) {
            this->cnodes[node].len_arr.emplace_back((mask & len) != 0);
            mask >>= 1;
        }

        return;
    }

    void encode(bits &bit_array, size_type x, int len) {
        for (int i = len - 1; i >= 0; i--) {
            bit_array.emplace_back((x >> i) & 1L);
        }
    }

    void encode_gamma(bits &bit_array, size_type x) {
        x++;
        assert(x >= 0);
        int len = this->get_significent_bit(x);
        this->encode(bit_array, 1, len + 1);
        this->encode(bit_array, x, len);
    }

    void encode_zeta(bits &bit_array, size_type x) {
        if (this->zeta_k == 1) {
            encode_gamma(bit_array, x);
        } else {
            x++;
            assert(x >= 0);
            int len = this->get_significent_bit(x);
            int h = len / this->zeta_k;
            this->encode(bit_array, 1, h + 1);
            this->encode(bit_array, x, (h + 1) * this->zeta_k);
        }
    }

    void append_gamma(bits &bit_array, size_type x) {
        if (x < this->PRE_ENCODE_NUM) {
            bit_array.insert(bit_array.end(), this->gamma_code[x].begin(), this->gamma_code[x].end());
        } else {
            encode_gamma(bit_array, x);
        }
    }

    void append_zeta(bits &bit_array, size_type x) {
        if (x < this->PRE_ENCODE_NUM) {
            bit_array.insert(bit_array.end(), this->zeta_code[x].begin(), this->zeta_code[x].end());
        } else {
            encode_zeta(bit_array, x);
        }
    }

    void append_bit(bits &bit_array, size_type x, int max_bit) {
        auto bit_cnt = get_bit_num(x);
        for (int i = bit_cnt; i < max_bit; ++i) bit_array.emplace_back(0);
        if (x == 0) bit_array.emplace_back(0);
        else {
            size_type mask = 1ll << (bit_cnt - 1);
            while (mask) {
                bit_array.emplace_back((mask & x) != 0);
                mask >>= 1;
            }
        }
        return;
    }

    void append_signed_bit(bits &bit_array, size_type x, int max_bit) {
        bool neg = x < 0ll;
        if (x < 0ll) x = 0ll - x;
        auto bit_cnt = get_bit_num(x);
        for (int i = bit_cnt; i < max_bit; ++i) bit_array.emplace_back(0);
        if (x == 0) bit_array.emplace_back(0);
        else {
            size_type mask = 1ll << (bit_cnt - 1);
            while (mask) {
                bit_array.emplace_back((mask & x) != 0);
                mask >>= 1;
            }
        }
        bit_array.emplace_back(neg);
        return;
    }

    void compress(std::string encode_type) {
        pre_encoding();

        this->cnodes.clear();
        this->cnodes.resize(this->num_node);

#pragma omp parallel for
        for (size_type i = 0; i < this->num_node; i++) {
            // if (i % 10000 == 0)  std::cout << i << std::endl;
            encode_node_s1(i);
        }

#pragma omp parallel for
        for (size_type i = 0; i < this->num_node; i++) {
            // if (i % 10000 == 0)  std::cout << i << std::endl;
            encode_node_s2(i);
        }

#pragma omp parallel for
        for (size_type i = 0; i < this->num_node; i++) {
            // if (i % 10000 == 0)  std::cout << i << std::endl;
            encode_offset(i);
        }
    }

    void pre_encoding() {
        this->gamma_code.clear();
        this->gamma_code.resize(this->PRE_ENCODE_NUM);

        this->zeta_code.clear();
        this->zeta_code.resize(this->PRE_ENCODE_NUM);

#pragma omp parallel for
        for (size_type i = 0; i < this->PRE_ENCODE_NUM; i++) {
            // pre-encode gamma
            encode_gamma(this->gamma_code[i], i);

            // pre-encode zeta
            if (this->zeta_k == 1) {
                this->zeta_code[i] = this->gamma_code[i];
            } else {
                encode_zeta(this->zeta_code[i], i);
            }
        }

        // for (size_type i = 0; i < this->PRE_ENCODE_NUM; i++) {
        //     std::cout << i << ' ';
        //     for (auto j : gamma_code[i]) std::cout << j;
        //     std::cout << ' ';
        //     for (auto j : zeta_code[i]) std::cout << j;
        //     std::cout << std::endl;
        // }
    }

    int get_significent_bit(size_type x) {
        assert(x > 0);
        int ret = 0;
        while (x > 1) x >>= 1, ret++;
        return ret;
    }


    void set_zeta_k(int _zeta_k) {
        Compressor::zeta_k = _zeta_k;
    }

};

#endif /* CGR_COMPRESSOR_HPP */