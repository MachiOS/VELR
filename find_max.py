
import pandas as pd
import numpy as np

import sys

args = sys.argv

directory = args[1]

step = args[2]

'''
directory = '/media/iec/Seagate Expansion Drive/mnist_classifier/step400_seed1001_data10/min_prob'

step = 4000


python find_max.py /media/iec/Seagate\ Expansion\ Drive/mnist_classifier/step400_seed1001_data10/min_prob 4000

python find_max.py /media/iec/Seagate\ Expansion\ Drive/mnist_classifier/step400_seed1001_data10/min_prob_consensus 4000

python find_max.py /media/iec/Seagate\ Expansion\ Drive/cifar_classifier/seed1001_data100/min_prob 4000

python find_max.py /media/iec/Seagate\ Expansion\ Drive/cifar_classifier/seed1001_data100/min_prob_consensus 4000

python find_max.py /media/iec/Seagate\ Expansion\ Drive/cifar_classifier/seed1001_data1500/mean_prob 9000

python find_max.py /media/iec/Expansion18/cifar_classifier/seed2002_data1500/mean_prob 9000

python find_max.py /media/santos/Expansion18/cifar_classifier/seed2002_data5000/results_0_100/min_prob 9000

'''


def find_max(min_prob_file, gt_label_file, output_dir, step, num_class=10):

    max_prob_output = output_dir + '/test_prob_max_' + str(step) + '.csv'
    accuracy_output = output_dir + \
        '/accuracy_consensus_hm_beta' + str(step) + '.csv'

    # ==========Find max probabiltiy and its class====================================================
    min_probs_df = pd.read_csv(min_prob_file)
    gt_label_df = pd.read_csv(gt_label_file)

    column_list = [str(i) for i in range(num_class)]
    min_probs_df['max_prob'] = min_probs_df[column_list].max(axis=1)
    min_probs_df['max_cls'] = pd.to_numeric(
        min_probs_df[column_list].idxmax(axis=1))

    maxprob_output_df = pd.concat([min_probs_df, gt_label_df['label']], axis=1)

    maxprob_output_df.astype({'max_cls': 'int32', 'label': 'int32'})
    # print(maxprob_output_df.dtypes)

    # print(np.where(maxprob_output_df['max_cls'] == maxprob_output_df['label']))
    maxprob_output_df['isCorrect'] = np.where(
        maxprob_output_df['max_cls'] == maxprob_output_df['label'], 1, 0)

    threshold_list = [num/100 for num in range(5, 100, 5)]

    for thld in threshold_list:
        column_name = 'thld_' + str(thld)
        maxprob_output_df[column_name] = np.where(
            maxprob_output_df['max_prob'] > thld, True, False)

    maxprob_output_df.to_csv(max_prob_output, index=False)

    # ==== Compute accruacy ====================================================================

    accuracy_df = pd.DataFrame(columns=['threshold', 'num_correct', 'total_pred', 'accuracy',
                               'harmonic_mean_alpha', 'harmonic_mean_beta', 'normalized_hm_beta', 'shifted_beta'])

    accuracy_full = maxprob_output_df['isCorrect'].sum()
    # total_pred = len(maxprob_output_df)
    num_alldata = len(maxprob_output_df)
    accruacy = accuracy_full / num_alldata

    tp_rate = accruacy

    harmonic_mean_alpha = 2 * (accruacy * 1.0) / (accruacy + 1.0)

    harmonic_mean_beta = 2 * accruacy * tp_rate / (accruacy + tp_rate)

    upper_bound_hm_beta = 2 * accuracy_full / (accuracy_full + num_alldata)

    noramlized_hm_beta = harmonic_mean_beta / upper_bound_hm_beta

    baseline_noramlized_hm_beta = noramlized_hm_beta

    dif_from_baseline_noramlized_hm_beta = 0
    max_dif = 1 - baseline_noramlized_hm_beta
    min_dif = -baseline_noramlized_hm_beta

    shifted_dif_from_baseline_noramlized_hm_beta = baseline_noramlized_hm_beta

    accuracy_df.loc[0] = [0, accuracy_full, num_alldata, accruacy, harmonic_mean_alpha,
                          harmonic_mean_beta, noramlized_hm_beta, shifted_dif_from_baseline_noramlized_hm_beta]

    for i, thld in enumerate(threshold_list):

        #  tp_rate is a recall like measure (#TP / total data) when we condier all data are "positives"

        column_name = 'thld_' + str(thld)
        accruacy_consensus = maxprob_output_df.loc[maxprob_output_df[column_name] == True, 'isCorrect'].sum(
        )

        total_pred = len(
            maxprob_output_df[maxprob_output_df[column_name] == True])

        percent_pred = total_pred / num_alldata
        tp_rate = accruacy_consensus / num_alldata

        if total_pred > 0:
            accruacy = accruacy_consensus / total_pred

            harmonic_mean_alpha = 2 * \
                (accruacy * percent_pred) / (accruacy + percent_pred)
            harmonic_mean_beta = 2 * accruacy * tp_rate / (accruacy + tp_rate)

            # same as 2*accruacy_consensus / (total_pred + num_alldata)

            noramlized_hm_beta = harmonic_mean_beta / upper_bound_hm_beta
            dif_from_baseline_noramlized_hm_beta = noramlized_hm_beta - \
                baseline_noramlized_hm_beta
            shifted_dif_from_baseline_noramlized_hm_beta = (
                dif_from_baseline_noramlized_hm_beta - min_dif) / (max_dif - min_dif)

        else:
            accruacy = 0
            harmonic_mean_alpha = 0
            harmonic_mean_beta = 0
            noramlized_hm_beta = 0
            # dif_from_baseline_noramlized_hm_beta = min_dif
            shifted_dif_from_baseline_noramlized_hm_beta = 0

        accuracy_df.loc[i+1] = [thld, accruacy_consensus, total_pred, accruacy, harmonic_mean_alpha,
                                harmonic_mean_beta, noramlized_hm_beta, shifted_dif_from_baseline_noramlized_hm_beta]

    # === compute shifted_normalized difference==========

    accuracy_df.to_csv(accuracy_output, index=False)


def main():

    num_class = 100
    # num_class = 100 # CIFAR-100

    if "mean_prob" in directory:
        min_prob_file = directory + '/test_prob_mean_all_' + str(step) + '.csv'
    else:
        min_prob_file = directory + '/test_prob_min_' + str(step) + '.csv'
    # min_prob_file = directory + '/test_prob_mean_all_'+ str(step) +'.csv'

    gt_label_file = directory + f'/test_label{step}.csv'

    find_max(min_prob_file, gt_label_file, directory, step, num_class)


if __name__ == '__main__':
    main()
