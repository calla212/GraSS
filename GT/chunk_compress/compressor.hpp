#ifndef CGR_COMPRESSOR_HPP
#define CGR_COMPRESSOR_HPP

#include <string>
#include <iostream>
#include <cassert>
#include <algorithm>
#include <vector>
#include <cmath>
#include <deque>

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
        size_type block_num, block_len, om_num, first_pos;
        int head_bit, body_bit;

        std::vector<size_type> head_node;
        std::vector<size_type> body_node;

        bits bit_arr;
        std::deque<bool> len_arr;

        void set_block_num(void) {
            size_type nsqrt = sqrt(outd);

            if (nsqrt * nsqrt == outd) {
                om_num = 0;
                block_num = block_len = nsqrt;
            }
            else {
                block_num = nsqrt + 1;
                block_len = outd / block_num;
                om_num = outd - block_num * block_len;
            }
            return;
        }

        void init(size_type _node, size_type _outd) {
            node = _node;
            outd = _outd;
            set_block_num();
            head_bit = body_bit = 0;
            first_pos = -1;
            head_node.clear();
            body_node.clear();
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
                    if (this->edges.back().second == v) continue;
                    assert(this->edges.back().second < v);
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

        this->write_bit_array(of_graph);
        fclose(of_graph);
    
        FILE *of_offset = fopen((dir_path + ".offset").c_str(), "w");

        if (of_offset == 0) {
            std::cout << "graph file cannot create!" << std::endl;
            abort();
        }

        this->write_offset3(of_offset);
        fclose(of_offset);

        return true;
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

    void write_bit_array(FILE* &of) {
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

    void encode_node_s1(const size_type node) {
        auto &cnode = this->cnodes[node];
        auto &neighbors = this->adjlist[node];

        cnode.init(node, neighbors.size());
        // assert(cnode.om_num >= 0);
        // if (cnode.om_num >= cnode.block_num) {
        //     std::cout << cnode.outd << " " << cnode.om_num << " " << cnode.block_num << "*" << cnode.block_len << std::endl;
        // }
        // if (cnode.outd) assert(cnode.om_num < cnode.block_num);
        // assert(cnode.outd == cnode.block_len * cnode.block_num + cnode.om_num);

        size_type cur_id = 0;
        for (size_type block_id = 0; block_id < cnode.block_num; ++block_id) {
            auto cur = neighbors[cur_id] - node;
            // std::cout << node << "[" << neighbors[cur_id] << "] " << block_id << "--" << cnode.block_num << " " << cur_id << std::endl;
            if (cur < 0) {
                cnode.first_pos = block_id;
                cur = 0 - cur;
            }
            --cur;
            cnode.head_node.emplace_back(cur);
            cnode.head_bit = std::max(cnode.head_bit, get_bit_num(cur));

            ++cur_id;
            for (int i = 1; i < cnode.block_len + (block_id < cnode.om_num) && cur_id < cnode.outd; ++i, ++cur_id) {
                cur = neighbors[cur_id] - neighbors[cur_id - 1] - 1;
                // std::cout << node << "[" << neighbors[cur_id] << "] " << block_id << "--" << cnode.block_num << " " << i << "==" << cnode.block_len << " " << cur_id <<" ->" << cur << std::endl;
                cnode.body_node.emplace_back(cur);
                cnode.body_bit = std::max(cnode.body_bit, get_bit_num(cur));
            }
        }
        ++cnode.first_pos;

        assert(cur_id == cnode.outd);

        return;
    }

    void set_max_bit(bool is_set) {
        this->max_head_bit = this->max_body_bit = -1;

        if (is_set) {
            for (auto cnode : this->cnodes) {
                max_head_bit = std::max(max_head_bit, cnode.head_bit);
                max_body_bit = std::max(max_body_bit, cnode.body_bit);
            }
        }

        return;
    }
    
    void encode_node_s2(const size_type node) {

        auto &cnode = this->cnodes[node];

        // std::cout << node << " " << cnode.outd << " " << cnode.head_bit << "(" << max_body_bit << ") " << cnode.body_bit << "(" << max_body_bit << ") " << cnode.first_pos << std::endl;
        // std::cout << cnode.block_num << "*" << cnode.block_len << "--" << cnode.lst_len << std::endl;

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
        if (max_body_bit == -1) {
            append_gamma(cnode.bit_arr, cnode.head_bit);
            append_gamma(cnode.bit_arr, cnode.body_bit);
        }
        append_gamma(cnode.bit_arr, cnode.first_pos);
#endif

        int this_head_bit = cnode.head_bit, this_body_bit = cnode.body_bit;
        if (max_head_bit != -1) this_head_bit = max_head_bit;
        if (max_body_bit != -1) this_body_bit = max_body_bit;

#ifdef GATHER_HEAD
        for (auto cur : cnode.head_node) append_bit(cnode.bit_arr, cur, this_head_bit);
        for (auto cur : cnode.body_node) append_bit(cnode.bit_arr, cur, this_body_bit);
#else
        auto &neighbors = this->adjlist[node];
        size_type cur_id = 0;
        for (size_type block_id = 0; block_id < cnode.block_num; ++block_id) {
            auto cur = neighbors[cur_id] - node;
            if (cur < 0) {
                cnode.first_pos = block_id;
                cur = 0 - cur;
            }
            append_bit(cnode.bit_arr, cur - 1, this_head_bit);

            ++cur_id;
            for (int i = 1; i < cnode.block_len + (block_id < cnode.om_num) && cur_id < cnode.outd; ++i, ++cur_id) {
                cur = neighbors[cur_id] - neighbors[cur_id - 1] - 1;
                append_bit(cnode.bit_arr, cur, this_body_bit);
            }
        }

        assert(cur_id == cnode.outd);
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
            size_type mask = 1 << (bit_cnt - 1);
            while (mask) {
                bit_array.emplace_back((mask & x) != 0);
                mask >>= 1;
            }
        }
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

        set_max_bit(encode_type == "head_bit");

#pragma omp parallel for
        for (size_type i = 0; i < this->num_node; i++) {
            // if (i % 10000 == 0)  std::cout << "S2 " << i << std::endl;
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