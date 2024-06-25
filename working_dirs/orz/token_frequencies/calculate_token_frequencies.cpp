/**
* Although we store counts of tokens saperately for memorized and non-memorized tokens, we can combine them to get the total counts of each token
*/

#include<bits/stdc++.h>
using namespace std;

// Iterates over the pile and counts the number of times each token is memorized and not memorized
// Saves the results to a json file
int run_pile(vector<int>& indicies){
    vector<long long> memorized = vector<long long>(60000, 0);
    vector<long long> non_memorized = vector<long long>(60000, 0);

    ifstream ios;
    ios.open("/scratch/pile/standard/document.bin", ios::binary);
    // Each sequence has 2049 tokens, every iteration during training has a global batch size of 1024
    int length = 2049*1024*16;
    uint16_t* buffer = new uint16_t[length];

    int curr = 0;
    printf("Starting Iteration\n");
    for (int i = 0;i < 143000*1024; i += 1024*16){
        ios.read((char*)buffer, length*sizeof(uint16_t));
        for(int j = 0;j < 1024*16;j++){
            bool is_mem = false;
            if (i + j >= 143000*1024){
                break;
            }
            if(indicies[curr] == (i + j)){
                is_mem = true;
                curr++;

                if(curr == indicies.size()){
                    is_mem = false;
                    curr = 0;
                }
            }
            else{
                is_mem = false;
            }

            for(int k = 0;k < 2049;k++){
                int data = buffer[j*2049+k];
                if(is_mem){
                    memorized[data]++;
                }
                else{
                    non_memorized[data]++;
                }
            }
        }
        printf("Finished Iteration %d\n", i/1024);
    }
    
    // Save the results to a json file
    ofstream savefile;
    savefile.open("/fsx/orz/semantic-memorization/working_dirs/orz/standard_memorized_frequencies.json");
    savefile << "{\n";
    savefile << "\"memorized\": [";
    for(int i = 0;i < 60000;i++){
        savefile << memorized[i];
        if(i != 59999){
            savefile << ", ";
        }
    }
    savefile << "],\n";
    savefile << "\"non_memorized\": [";
    for(int i = 0;i < 60000;i++){
        savefile << non_memorized[i];
        if(i != 59999){
            savefile << ", ";
        }
    }
    savefile << "]\n";
    savefile << "}";
    savefile.close();
    return 0;
}


// Loads the indicies of the memorized tokens, these were stored on a text file, you can replicate them by storing the indicies of the memorized tokens (with memorization score of 1)
// You can find list of memorized indicies at https://huggingface.co/datasets/EleutherAI/pythia-memorized-evals
// Note that we are only using duped.12b and deduped.12b splits
// on a text file (mem_standard.txt) and then loading them here
int load_indicies(vector<int>& data){
    fstream ios;
    ios.open("/fsx/orz/memorization-evals/memorized_evals/mem_standard.txt");
    int index;
    while (ios >> index){
        data.push_back(index);
    }
    sort(data.begin(), data.end());
    ios.close();
    return 0;
}

int main(){
    vector<int> indicies;
    load_indicies(indicies);
    run_pile(indicies);
    return 0;
}