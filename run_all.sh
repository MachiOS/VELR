
# num_parallel=3

a_flag=''
d_flag=''
b_flag=''
n_flag=''
e_flag=''
i_flag=''
t_flag=''
s_flag=''
m_flag=''
j_flag=''
k_flag=''

print_usage() {
  printf "Usage: ..."
}

while getopts 'd:a:b:n:e:i:t:s:m:j:k:p:' flag; do
  case "${flag}" in
    d) d_flag="${OPTARG}" ;; # directory to store in 
    a) a_flag="${OPTARG}" ;; # dataset directory
    b) b_flag="${OPTARG}" ;; # batch size
    n) n_flag="${OPTARG}" ;; # num iters
    e) e_flag="${OPTARG}" ;; # eval_size
    i) i_flag="${OPTARG}" ;; # eval every n iter
    t) t_flag="${OPTARG}" ;; # num_data
    s) s_flag="${OPTARG}" ;; # dataset 
    m) m_flag="${OPTARG}" ;; # model 
    j) j_flag="${OPTARG}" ;; # start model number 
    k) k_flag="${OPTARG}" ;; # end model number
    p) p_flag="${OPTARG}" ;; # num of parallel processes to run
    *) print_usage
       exit 1 ;;
  esac
done

num_parallel=$p_flag
num_models=$(($k_flag-$j_flag))
one_step_above_path=${d_flag%/*}
one_step_above_path=${one_step_above_path%/*}
echo "$one_step_above_path"
echo $num_models

# SAMPLE
# # python run_classifier_cifar.py -dir_path $d_flag -data_path $a_flag -batch_size $b_flag \
#  # -num_train_cls $n_flag -eval_size $e_flag -eval_every_n_iter $i_flag -num_data $t_flag \
#  # -dataset $s_flag -model_dir $m_flag -model_start $j_flag -model_end $k_flag 

# run training in parallel, keep all in -> M_0.

job(){
  python run_classifier_cifar.py -dir_path $d_flag -data_path $a_flag -batch_size $b_flag \
  -num_train_cls $n_flag -eval_size $e_flag -eval_every_n_iter $i_flag -num_data $t_flag \
  -dataset $s_flag -model_dir $m_flag -model_start $1 -model_end $2 
  echo "Done training $1 to $2"
}

per_split=$(($num_models/$num_parallel))

for ((i = 0 ; i < $num_parallel ; i++)); do
    if [[ "$i" == "$(($num_parallel-1))" ]]; then
        # echo "here"
        k_val=$num_models
    else 
        k_val=$((($i+1)*$per_split))
    fi

    j_val=$(($i*$per_split))
    # echo "j_val $j_val"
    # echo "k_val $k_val"
    echo "Starting training for $j_val to $k_val"
    job $j_val $k_val &

    pids="$pids $!"

    echo "Wait for 5 sec to initialize"
    sleep 5
done

for pid in $pids; do
    echo "waiting to finish training"
    wait $pid || let "RESULT=1"
done

if [ "$RESULT" == "1" ]; then
    exit 1
fi

echo "DONE Training all"

python get_minprob_mean_pool.py $one_step_above_path $n_flag $num_models

# # rename_to_path=$one_step_above_path"/M_100"
# # mv $d_flag $rename_to_path

# # python get_minprob_GMM_v2.py $one_step_above_path $n_flag $num_models 

python get_minprob_mnist_v2.py $one_step_above_path $n_flag $num_models

test_label_file="test_label"$n_flag".csv"

# # min_prob_folder=""
# # echo $d_flag"model"$j_flag"/"$test_label_file 
# # echo $one_step_above_path"/min_prob/"$test_label_file

cp $d_flag"model"$j_flag"/"$test_label_file $one_step_above_path"/min_prob/"$test_label_file
cp $d_flag"model"$j_flag"/"$test_label_file $one_step_above_path"/min_prob_normal/"$test_label_file

# # echo $d_flag"model"$j_flag"/"$test_label_file $one_step_above_path"/min_prob_normal/"$test_label_file

# python find_max.py $one_step_above_path"/min_prob" $n_flag
python find_max.py $one_step_above_path"/min_prob_normal" $n_flag
python find_max.py $one_step_above_path"/results_$j_flag""_"$k_flag"/mean_prob" $n_flag

# # mkdir $one_step_above_path"/min_prob/model0_100"

# # rename_again_path=$one_step_above_path"/M_0"
# # mv $rename_to_path $rename_again_path

# # GMM 
# python draw_histgram_v2.py $one_step_above_path $n_flag $num_models $one_step_above_path"/min_prob"
# python draw_histgram_selective.py $one_step_above_path $n_flag $num_models $one_step_above_path"/min_prob"

python get_scores.py \
-save_dir $one_step_above_path \
-normal_file_path $one_step_above_path"/min_prob_normal/test_prob_max_"$n_flag.csv \
-gmm_file_path $one_step_above_path"/min_prob/test_prob_max_"$n_flag".csv" \
-uniform_file_path $one_step_above_path"/results_"$j_flag"_"$k_flag"/mean_prob/test_prob_max_"$n_flag".csv" \
-start_model_num $j_flag -end_model_num $k_flag -num_classes 100

