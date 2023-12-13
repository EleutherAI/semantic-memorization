#include<bits/stdc++.h>
#include<mpi.h>
#include<filesystem>
#include "cnpy/cnpy.h"

using namespace std;

const string DS_TYPE = "standard";
const string PILE_BIN_LOCATION = "/scratch/pile/";
/**
* Relative location of dataset. The script assumes true location to be
* {PILE_BIN_LOCATION}/{DS_TYPE}/document.bin
*/

const string DATA_DIR = "../results/";
const int NGRAMS = 32;
const int OFFSET_START_IDX = 32;

void print_progress(string prefix, float progress){
    cout << "\33[2k\r" << prefix << " " << setprecision(4) << progress;
    cout.flush();
}

void get_arr_from_pos(pair<int64_t, int32_t> pos, uint16_t* buff, ifstream& pilein){
    uint64_t fileoff;
    fileoff = pos.first*2049;
    fileoff += pos.second;
    fileoff *= 2;
    pilein.seekg(fileoff);
    pilein.read((char*)buff, NGRAMS*2);
    while(pilein.gcount() != NGRAMS*2){
        pilein.seekg(fileoff);
        pilein.read((char*)buff, NGRAMS*2);
    }
}

bool compare_arrays(uint16_t* arr1, pair<int64_t, int32_t> pos2, ifstream& pilein, uint16_t* arr2){
    get_arr_from_pos(pos2, arr2, pilein);
    for(int i = 0;i < NGRAMS; i++){
        if(arr1[i] != arr2[i]){
            return false;
        }
    }
    return true;
}

void get_exact_duplicates(
    vector<vector<pair<uint32_t,uint16_t>>>& hashCounts,
    unordered_map<uint32_t, uint32_t>& offsets,
    ifstream& pilein,
    map<uint32_t, uint32_t>& true_counts,
    int process_Rank
){
    int size = offsets.size();
    cout << "Calculating Exact duplicates " << endl;
    string prefix = "[RANK: " + to_string(process_Rank) + "]: ";  
    uint16_t* curr_arr = new uint16_t[64];
    uint16_t* arr2 = new uint16_t[64];
    
    
    int currpos = 0;
    float next_prog = 0.0;
    for(auto& x:offsets){
        int i = x.first;
        auto hash = x.second;
        if(hashCounts[hash].size() != 0){
            int count = 0;
            get_arr_from_pos(make_pair(i, OFFSET_START_IDX), curr_arr, pilein);
            for(auto& potent_dup_pos:hashCounts[hash]){
                if(compare_arrays(curr_arr, potent_dup_pos, pilein, arr2))
                    count += 1;
            }
            true_counts[i] = count;
        }
        else{
            true_counts[i] = 1;
        }

        float progress = ((float)currpos) / size;
        progress *= 100;
        if(next_prog < progress){
            print_progress(prefix, next_prog);
            next_prog += 0.5;
        }
        currpos += 1;
    }

}

void load_data(
    vector<vector<pair<uint32_t,uint16_t>>>& hashCounts, 
    unordered_map<uint32_t, uint32_t>& offsets,
    int process_Rank, int size_Of_Cluster
){
    cnpy::npz_t data = cnpy::npz_load(DATA_DIR + DS_TYPE + ".npz");
    uint32_t size = data["hash"].shape[0];
    uint64_t* Hashes = data["hash"].data<uint64_t>();
    uint16_t* Offsets = data["offset"].data<uint16_t>();
    uint32_t* Indicies = data["index"].data<uint32_t>();


    cout << "Loading Offsets" << endl;
    double next_prog = 0.0;
    string prefix = "[RANK: " + to_string(process_Rank) + "]: ";    

    vector<uint8_t> curRankHashes(hashCounts.size(), 0);
    
    for(uint32_t i = 0;i < size;i++){
        pair<uint32_t, uint16_t> pos = make_pair(Indicies[i], Offsets[i]);
        if(pos.second == OFFSET_START_IDX){
            if(pos.first % size_Of_Cluster == process_Rank){
                offsets[pos.first] = Hashes[i]%hashCounts.size();
                curRankHashes[Hashes[i]%hashCounts.size()] = true;
            }
        }
        float progress = ((float)i) / size;
        progress *= 100;
        if(next_prog < progress){
            print_progress(prefix, next_prog);
            next_prog += 0.5;
        }
    }


    cout << "Loading Hash Counts " << endl;
    next_prog = 0.0;
    for(uint32_t i = 0;i < size;i++){
        if(curRankHashes[Hashes[i] % hashCounts.size()]){
            pair<uint32_t, uint16_t> pos = make_pair(Indicies[i], Offsets[i]);
            hashCounts[Hashes[i]%hashCounts.size()].push_back(pos);
        }

        float progress = ((float)i) / size;
        progress *= 100;
        if(next_prog < progress){
            print_progress(prefix, next_prog);
            next_prog += 0.5;
        }
    }

}

void combine_truecounts(int size_Of_Cluster){
    vector<uint32_t> indicies, counts;
    string save_path;
    for(int rank = 0;rank < size_Of_Cluster;rank++){
        save_path = "true_counts_" + DS_TYPE + to_string(rank) + ".npz";
        cnpy::npz_t data = cnpy::npz_load(save_path);

        uint32_t* idxs = data["indicies"].data<uint32_t>();
        uint32_t* cnts = data["counts"].data<uint32_t>();
        for(size_t i = 0;i < data["indicies"].shape[0];i++){
            indicies.push_back(idxs[i]);
            counts.push_back(cnts[i]);
        }
    }

    save_path = DATA_DIR + "true_counts_" + DS_TYPE + ".npz";
    cnpy::npz_save(save_path, "indicies", &indicies[0], {indicies.size(),}, "a");
    cnpy::npz_save(save_path, "counts", &counts[0], {counts.size(),}, "a");
}
void save_trueCounts(map<uint32_t, uint32_t>& trueCounts, int process_Rank){
    vector<uint32_t> indicies, counts;
    for(auto& x:trueCounts){
        indicies.push_back(x.first);
        counts.push_back(x.second);
    }
    
    string save_path = "true_counts_" + DS_TYPE +  to_string(process_Rank) + ".npz";
    cnpy::npz_save(save_path, "indicies", &indicies[0], {indicies.size(),}, "a");
    cnpy::npz_save(save_path, "counts", &counts[0], {counts.size(),}, "a");
}

int main(int argc, char**argv){
    int process_Rank, size_Of_Cluster, message_Item;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size_Of_Cluster);
    MPI_Comm_rank(MPI_COMM_WORLD, &process_Rank);

    vector<vector<pair<uint32_t,uint16_t>>> hashCounts(111191111); // PS: It's a prime, called as palindromic prime :)
    unordered_map<uint32_t, uint32_t> offsets;
    for(int rank = 0;rank < size_Of_Cluster;rank+=16){
        if(rank <= process_Rank && (rank + 16) > process_Rank){
            load_data(hashCounts, offsets, process_Rank, size_Of_Cluster);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    

    ifstream pilein;
    pilein.open(PILE_BIN_LOCATION + DS_TYPE + "/document.bin", ios::binary);
    map<uint32_t, uint32_t> trueCounts;
    get_exact_duplicates(hashCounts, offsets, pilein, trueCounts, process_Rank);

    MPI_Barrier(MPI_COMM_WORLD);
    save_trueCounts(trueCounts, process_Rank);
    MPI_Barrier(MPI_COMM_WORLD);
    trueCounts.clear();
    offsets.clear();
    hashCounts.clear();

    if(process_Rank == 0)
        combine_truecounts(size_Of_Cluster);

    MPI_Finalize();
}