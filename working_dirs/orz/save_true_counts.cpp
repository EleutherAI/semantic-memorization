#include<bits/stdc++.h>

#include <arrow/api.h>
#include <arrow/csv/api.h>
#include <arrow/io/api.h>
#include <arrow/ipc/api.h>
#include <parquet/arrow/reader.h>
#include <parquet/arrow/writer.h>

#include<mpi.h>

#include "tqdm.hpp"

using namespace std;

const string DS_TYPE = "standard";

void get_arr_from_pos(pair<int64_t, int32_t> pos, uint16_t* buff, ifstream& pilein){
    uint64_t fileoff;
    fileoff = pos.first*2049;
    fileoff += pos.second;
    fileoff *= 2;
    pilein.seekg(fileoff);
    pilein.read((char*)buff, 64*2);
    while(pilein.gcount() != 64*2){
        pilein.seekg(fileoff);
        pilein.read((char*)buff, 64*2);
    }
}

bool compare_arrays(uint16_t* arr1, pair<int64_t, int32_t> pos2, ifstream& pilein, uint16_t* arr2){
    get_arr_from_pos(pos2, arr2, pilein);
    for(int i = 0;i < 64; i++){
        if(arr1[i] != arr2[i]){
            return false;
        }
    }
    return true;
}

void get_exact_duplicates(
    vector<vector<pair<int64_t,int32_t>>>& hashCounts,
    unordered_map<int64_t, uint64_t>& zeroOffsets,
    ifstream& pilein,
    map<int64_t, int64_t>& true_counts,
    int process_Rank
){
    cout << "Calculating Exact duplicates " << zeroOffsets.size() << endl;
    auto iter = tq::tqdm(zeroOffsets);
    iter.set_prefix("[RANK: " + to_string(process_Rank) + "]: ");
    uint16_t* curr_arr = new uint16_t[64];
    uint16_t* arr2 = new uint16_t[64];
    for(auto x:iter){
        int i = x.first;
        auto hash = x.second;
        if(hashCounts[hash].size() != 0){
            int count = 0;
            get_arr_from_pos(make_pair(i, 0), curr_arr, pilein);
            for(auto& potent_dup_pos:hashCounts[hash]){
                if(compare_arrays(curr_arr, potent_dup_pos, pilein, arr2))
                    count += 1;
            }
            if(count == 0)
                count = 1;
            true_counts[i] = count;
        }
        else{
            true_counts[i] = 1;
        }
    }
}

void load_hashCounts(vector<vector<pair<int64_t,int32_t>>>& hashCounts, int process_Rank){
    ifstream in;
    in.open("hashCounts.txt");
    
    cout << "Loading Hashes" << endl;
    string text;
    int count = 0;
    while(getline(in, text))count++;
    in.close();
    in.open("hashCounts.txt");
    auto iter = tq::trange(count);
    iter.set_prefix("[RANK: " + to_string(process_Rank) + "]: ");
    for(int i:iter){
        uint64_t hash;
        in >> hash;
        hash = hash%hashCounts.size();
        int64_t size;
        in >> size;
        for(int i = 0;i < size;i++){
            pair<int64_t,int32_t> pos;
            in >> pos.first >> pos.second;
            hashCounts[hash].push_back(pos);
        }
    }
    cout << endl;
}

void load_zeroOffsets(
    unordered_map<int64_t, uint64_t>& zeroOffsets, 
    int64_t bucket_size,
    int process_Rank,
    int size_Of_Cluster
){
    ifstream in;
    in.open("zero_offsets.txt");
    cout << "Loading Zero Offsets" << endl;

    arrow::MemoryPool* pool = arrow::default_memory_pool();
    shared_ptr<arrow::io::RandomAccessFile> input;
    input = arrow::io::ReadableFile::Open("../" + DS_TYPE + "_counts.parquet").ValueOrDie();

    unique_ptr<parquet::arrow::FileReader> arrow_reader;
    auto status = parquet::arrow::OpenFile(input, pool, &arrow_reader);

    std::shared_ptr<arrow::Table> table;
    status = arrow_reader->ReadTable(&table);

    auto schema = table->schema();
    if(process_Rank == 0)
        cout << "Schema: " << schema->ToString() << endl;

    auto chunked_arrays = table->columns();
    auto indicies = static_pointer_cast<arrow::Int64Array>(chunked_arrays[0]->chunk(0));
    auto offsets = static_pointer_cast<arrow::Int32Array>(chunked_arrays[1]->chunk(0));
    auto hashes = static_pointer_cast<arrow::UInt64Array>(chunked_arrays[2]->chunk(0));
    auto counts = static_pointer_cast<arrow::Int64Array>(chunked_arrays[3]->chunk(0));

    auto iter = tq::trange(143000*1024);
    iter.set_prefix("[RANK: " + to_string(process_Rank) + "]: ");

    int elem_idx = 0;
    for(int i:iter){
        uint64_t hash;
        in >> hash;
        if(counts->Value(i) > 1){
            if(elem_idx % size_Of_Cluster == process_Rank)
                zeroOffsets[i] = hash%bucket_size;
            elem_idx += 1;
        }
    }
    cout << endl;
}

void save_trueCounts(map<int64_t, int64_t>& trueCounts, int process_Rank){
    ofstream csv;
    csv.open("true_counts_" + to_string(process_Rank) + ".csv");

    csv << "Index,Count\n";
    for(auto& x:trueCounts){
        csv << x.first << "," << x.second << "\n";
    }
    
}

int main(int argc, char**argv){
    int process_Rank, size_Of_Cluster, message_Item;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size_Of_Cluster);
    MPI_Comm_rank(MPI_COMM_WORLD, &process_Rank);

    vector<vector<pair<int64_t,int32_t>>> hashCounts(143000*1024);
    unordered_map<int64_t, uint64_t> zeroOffsets;
    load_zeroOffsets(zeroOffsets, hashCounts.size(), process_Rank, size_Of_Cluster);
    load_hashCounts(hashCounts, process_Rank);
    MPI_Barrier(MPI_COMM_WORLD);

    ifstream pilein;
    pilein.open("/scratch/pile/" + DS_TYPE + "/document.bin", ios::binary);
    map<int64_t, int64_t> trueCounts;
    get_exact_duplicates(hashCounts, zeroOffsets, pilein, trueCounts, process_Rank);

    MPI_Barrier(MPI_COMM_WORLD);
    save_trueCounts(trueCounts, process_Rank);

    MPI_Finalize();
}