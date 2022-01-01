#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <map>
using namespace std;

struct Edge {
    int u, v;

    bool operator==(const Edge i) {
        return (i.u == this->u) && (i.v == this->v);
    }
};

inline bool cmp(Edge a, Edge b) {
    if (a.u == b.u) return a.v < b.v;
    return a.u < b.u;
}

int main(int argc,char *argv[]) {
    if (argc < 4) {
        printf("incorrect arguments.\n");
        printf("<input_path> <output_path> <u/d> [<skip_line>]\n");
        abort();
    }

    int sl = 0;
    bool directed = argv[3][0] == 'd';
    if (argc == 5) sl = atoi(argv[4]);

    std::string input_path(argv[1]);
    std::string output_path(argv[2]);
    map<int, int> o2n;
    int tot = 0;
    Edge edge;
    vector<Edge> edges;
    ifstream fin;
    
    fin.open(input_path);
    printf("Skip %d Lines\n", sl);
    while (sl--) {
        char buf[500];
        fin.getline(buf, 500);
    }
    while (fin >> edge.u >> edge.v) {
        if (edge.u == edge.v) continue;
        if (o2n.count(edge.u) == 0) o2n[edge.u] = tot++;
        if (o2n.count(edge.v) == 0) o2n[edge.v] = tot++;
        edge.u = o2n[edge.u];
        edge.v = o2n[edge.v];
        if (!directed) {
            if (edge.u > edge.v) swap(edge.u, edge.v);
        }
        edges.emplace_back(edge);
    }

    fin.close();

    sort(edges.begin(), edges.end(), cmp);
    edges.erase(unique(edges.begin(), edges.end()), edges.end());
    if (!directed) {
        // edges.erase(unique(edges.begin(), edges.end()), edges.end());
        int len = edges.size();
        for (int i = 0; i < len; ++i) {
            edge = edges[i];
            swap(edge.u, edge.v);
            edges.emplace_back(edge);
        }
        sort(edges.begin(), edges.end(), cmp);
    }

    ofstream fout;
    fout.open(output_path);
    for (auto e : edges) fout << e.u << ' ' << e.v << '\n';
    fout.close();

    std::cout << "Graph has " << tot << " vertices, " << edges.size() << " edges.\n";
    
    return 0;
}
