#!/bin/bash

export LD_LIBRARY_PATH="/mnt/ssd-1/sai/miniconda3/lib"
seed=1024
step=30
# for end_pos in $(seq $step $step 810)
# do 
#     start_pos=$(($end_pos-$step))
#     echo $start_pos $end_pos
#     python model_training.py --taxonomy_search_start_index $start_pos --taxonomy_search_end_index $end_pos --run_id "run-s$seed-$start_pos-$end_pos" --sequence_duplication_threshold 6&
# done
# wait

python model_training.py --taxonomy_search_start_index $start_pos --taxonomy_search_end_index $end_pos --run_id "run-s$seed-t1" --sequence_duplication_threshold 1
python model_training.py --taxonomy_search_start_index $start_pos --taxonomy_search_end_index $end_pos --run_id "run-s$seed-t10" --sequence_duplication_threshold 11
# 

# start_pos=50
# end_pos=100
# python3 model_training.py --taxonomy_search_start_index $start_pos --taxonomy_search_end_index $end_pos --run_id "run-s$seed-$start_pos-$end_pos" --sequence_duplication_threshold 6&

# start_pos=100
# end_pos=150
# python3 model_training.py --taxonomy_search_start_index $start_pos --taxonomy_search_end_index $end_pos --run_id "run-s$seed-$start_pos-$end_pos" --sequence_duplication_threshold 6&

# start_pos=150
# end_pos=200
# python3 model_training.py --taxonomy_search_start_index $start_pos --taxonomy_search_end_index $end_pos --run_id "run-s$seed-$start_pos-$end_pos" --sequence_duplication_threshold 6&

# start_pos=200
# end_pos=250
# python3 model_training.py --taxonomy_search_start_index $start_pos --taxonomy_search_end_index $end_pos --run_id "run-s$seed-$start_pos-$end_pos" --sequence_duplication_threshold 6&

# start_pos=250
# end_pos=300
# python3 model_training.py --taxonomy_search_start_index $start_pos --taxonomy_search_end_index $end_pos --run_id "run-s$seed-$start_pos-$end_pos" --sequence_duplication_threshold 6&

# start_pos=300
# end_pos=350
# python3 model_training.py --taxonomy_search_start_index $start_pos --taxonomy_search_end_index $end_pos --run_id "run-s$seed-$start_pos-$end_pos" --sequence_duplication_threshold 6&

# start_pos=350
# end_pos=400
# python3 model_training.py --taxonomy_search_start_index $start_pos --taxonomy_search_end_index $end_pos --run_id "run-s$seed-$start_pos-$end_pos" --sequence_duplication_threshold 6&

# start_pos=400
# end_pos=450
# python3 model_training.py --taxonomy_search_start_index $start_pos --taxonomy_search_end_index $end_pos --run_id "run-s$seed-$start_pos-$end_pos" --sequence_duplication_threshold 6&

# start_pos=450
# end_pos=500
# python3 model_training.py --taxonomy_search_start_index $start_pos --taxonomy_search_end_index $end_pos --run_id "run-s$seed-$start_pos-$end_pos" --sequence_duplication_threshold 6&

# start_pos=500
# end_pos=550
# python3 model_training.py --taxonomy_search_start_index $start_pos --taxonomy_search_end_index $end_pos --run_id "run-s$seed-$start_pos-$end_pos" --sequence_duplication_threshold 6&

# start_pos=550
# end_pos=600
# python3 model_training.py --taxonomy_search_start_index $start_pos --taxonomy_search_end_index $end_pos --run_id "run-s$seed-$start_pos-$end_pos" --sequence_duplication_threshold 6&

# start_pos=600
# end_pos=650
# python3 model_training.py --taxonomy_search_start_index $start_pos --taxonomy_search_end_index $end_pos --run_id "run-s$seed-$start_pos-$end_pos" --sequence_duplication_threshold 6&

# start_pos=650
# end_pos=700
# python3 model_training.py --taxonomy_search_start_index $start_pos --taxonomy_search_end_index $end_pos --run_id "run-s$seed-$start_pos-$end_pos" --sequence_duplication_threshold 6&

# start_pos=750
# end_pos=800
# python3 model_training.py --taxonomy_search_start_index $start_pos --taxonomy_search_end_index $end_pos --run_id "run-s$seed-$start_pos-$end_pos" --sequence_duplication_threshold 6&

# start_pos=800
# end_pos=810
# python3 model_training.py --taxonomy_search_start_index $start_pos --taxonomy_search_end_index $end_pos --run_id "run-s$seed-$start_pos-$end_pos" --sequence_duplication_threshold 6

wait
