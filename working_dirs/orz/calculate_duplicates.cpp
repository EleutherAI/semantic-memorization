#include<bits/stdc++.h>
#include<mpi.h>
using namespace std;


const int MOD = 1e8 + 7;

// Util function that efficiently calculates a^b%MOD
// Refer https://cp-algorithms.com/algebra/binary-exp.html#implementation for more info
int binpow(int a, int b) {
    a %= MOD;
    int res = 1;
    while (b > 0) {
        if (b & 1)
            res = res * a % MOD;
        a = a * a % MOD;
        b >>= 1;
    }
    return res % MOD;
}

// Computes hash of a subsequence, taking 64 tokens from start position
int compute_hash(vector<int>& sequence, int start){
    int hash = 0;
    int position = 1;
    for(int i = start;i < start + 64;i++,position++){
        hash += binpow(sequence[i], position);
        hash %= MOD;
    }
    hash += MOD;
    return hash%MOD;
}

// Stores number of subsequence occurances in the bucket
void compute_duplicates(
    vector<set<pair<int,int>>>& bucket,
    set<int>& visited, 
    vector<vector<int>>& batch, 
    int batch_idx
){
    for(vector<int>& seq: batch){
        for(int i = 0; i < seq.size() - 64;i++){
            int hash = compute_hash(seq, i);

            bucket[hash].insert(make_pair(batch_idx, i));
            visited.insert(hash);
            batch_idx++;
        }
    } 
}


// Calculates number of duplicates in a batch and saves the results into a json file
void compute_over_batch(vector<vector<int>>& batch, int batch_idx){
    vector<set<pair<int,int>>> bucket(MOD + 10);
    set<int> visited;
    compute_duplicates(bucket, visited, batch, batch_idx);


    ofstream saveduplicates;
    saveduplicates.open("./duplicates/" + to_string(batch_idx) + ".json");
    saveduplicates << "{\n";

    bool isfirst = true;    
    for (auto hash:visited){
        if(isfirst) isfirst = false;
        else saveduplicates << ",\n";
        
        saveduplicates << "\"" << hash << "\":[";

        bool first_pos = true;
        for(pair<int,int> position:bucket[hash]){
            if(first_pos) first_pos = false;
            else {
                saveduplicates << ", ";
            }
            saveduplicates << "[" << position.first << ", " << position.second << "]";
        }
        saveduplicates << "]";
    }

    saveduplicates << "\n}";

}

// Iterates over the dataset and generates batches for computation.
// Processes are synced after every batch
void iterate_over_dataset(int start_idx, int end_idx){
    ifstream ios;
    ios.open("/scratch/pile/standard/document.bin", ios::binary);

    // Each sequence has 2049 tokens, every iteration during training 
    // has a global batch size of 1024
    int length = 2049*1024;
    uint16_t* buffer = new uint16_t[length];

    int curr = 0;

    // Only print messages on rank 0
    if(start_idx == 0)
        cout << "Starting Iteration\n";
    ios.seekg(start_idx*1024*2049*sizeof(uint16_t));
    for (int i = start_idx*1024;i < end_idx*1024; i += 1024){
        ios.read((char*)buffer, length*sizeof(uint16_t));

        // Iterating over the batch
        vector<vector<int>> batch;
        for(int j = 0;j < 1024;j++){
            if (i + j >= end_idx*1024){
                cout << "ERROR. Out Of Index! This should not happen!";
                return;
            }
            vector<int> sequence;
            for(int k = 0;k < 2049;k++){
                sequence.push_back(buffer[j*2049+k]);
            }
            batch.push_back(sequence);
        }

        compute_over_batch(batch, i);

        if(start_idx == 0)
            cout << "Completed Iteration: " << i / 1024 << "/" << end_idx << endl;
        MPI_Barrier(MPI_COMM_WORLD);
    }
}

int main(int argc, char**argv){
    int process_Rank, size_Of_Cluster, message_Item;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size_Of_Cluster);
    MPI_Comm_rank(MPI_COMM_WORLD, &process_Rank);

    if(143000 % size_Of_Cluster != 0){
        cout << "Invalid number of processes. Make sure that the number of processes are divisible";
        cout << " by 143000" << endl;
        return -1;
    }
    int start_idx = (process_Rank*143000)/size_Of_Cluster;
    int end_idx = start_idx + (143000/size_Of_Cluster);

    iterate_over_dataset(start_idx, end_idx);
}