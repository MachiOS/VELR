# VELR

## Run all

```
bash run_all.sh -d < directory to store trained outputs in> -a <dataset directory> -b <batch_size> -n <Number of training steps> -e <evaluation data size> -i <evaluate every i steps> -t <number of labeld samples> -s <name of the dataset -m model -j <start model number> -k <end model number> -p <number of paralels process to run>
```
Runs files in the order:
- run_classifier_cifar.py
- get_minprob_mean_pool.py 
- get_minprob_GMM_v2.py or get_minprob_mnist_v2.py (change the script accordingly)
- find_max.py (for both GMM and uniform)
- draw_histgram_v2.py
- draw_histgram_selective.py
- get_scores.py 


## Example
```
bash run_all.sh -d cifar10_classifier/labeled_data500/ -a datasets/cifar/data -b 32 -n 9000 -e 10000 -i 2000 -t 500 -s CIFAR10 -m model -j 0 -k 20 -p 3
```

```
bash run_all.sh -d cifar100_classifier/labeled_data500/ -a datasets/cifar/data -b 32 -n 2000 -e 10000 -i 2000 -t 5000 -s CIFAR100 -m model -j 0 -k 20 -p 3
```