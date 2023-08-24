#include<bits/stdc++.h>

#include <arrow/api.h>
#include <arrow/csv/api.h>
#include <arrow/io/api.h>
#include <arrow/ipc/api.h>
#include <parquet/arrow/reader.h>
#include <parquet/arrow/writer.h>

#include "tqdm.hpp"

using namespace std;

const string DS_TYPE = "standard";

void get_approx_duplicates(
    string ds_type,
    unordered_map<uint64_t, vector<pair<int64_t,int32_t>>>& hashCounts,
    vector<uint64_t>& zeroOffsets
){
    arrow::MemoryPool* pool = arrow::default_memory_pool();
    std::shared_ptr<arrow::io::RandomAccessFile> input;
    input = arrow::io::ReadableFile::Open("../" + ds_type + ".parquet").ValueOrDie();

    std::unique_ptr<parquet::arrow::FileReader> arrow_reader;
    auto status = parquet::arrow::OpenFile(input, pool, &arrow_reader);

    std::shared_ptr<arrow::Table> table;
    status = arrow_reader->ReadTable(&table);

    auto schema = table->schema();
    cout << "Schema: " << schema->ToString() << endl;

    auto chunked_arrays = table->columns();
    auto indicies = static_pointer_cast<arrow::Int64Array>(chunked_arrays[0]->chunk(0));
    auto offsets = static_pointer_cast<arrow::Int32Array>(chunked_arrays[1]->chunk(0));
    auto hashes = static_pointer_cast<arrow::UInt64Array>(chunked_arrays[2]->chunk(0));

    cout << "Iterating over parquet file: " << indicies->length() << endl;
    for(int i:tq::trange(indicies->length())){
        int64_t index = indicies->Value(i);
        int32_t offset = offsets->Value(i);
        uint64_t hash = hashes->Value(i);
        hashCounts[hash].push_back(make_pair(index, offset));

        if(offset == 0){
            zeroOffsets[index] =  hash;
        }
    }

    vector<uint64_t> erasehashes;
    for(auto& count:hashCounts){
        if (count.second.size() < 2){
            erasehashes.push_back(count.first);
        }
    }

    for(auto& hash:erasehashes){
        hashCounts.erase(hash);
    }

}

void save_hashcounts(unordered_map<uint64_t, vector<pair<int64_t,int32_t>>>& hashCounts){
    ofstream out;
    out.open("hashCounts.txt");
    for(auto& counts:hashCounts){
        out << counts.first << " " << counts.second.size() << " ";
        for(auto& pos:counts.second){
            out << pos.first << " " << pos.second << " ";
        }
        out << "\n";
    }
    out.close();
}
void save_zero_offsets(vector<uint64_t>& zeroOffsets){
    ofstream out;
    out.open("zero_offsets.txt");
    for(auto& x:zeroOffsets){
        out << x << "\n";
    }
    out.close();
}

int main(){
    unordered_map<uint64_t, vector<pair<int64_t,int32_t>>> hashCounts;
    vector<uint64_t> zeroOffsets(143000*1024);
    get_approx_duplicates(DS_TYPE, hashCounts, zeroOffsets);
    save_hashcounts(hashCounts);
    save_zero_offsets(zeroOffsets);
    return 0;
}