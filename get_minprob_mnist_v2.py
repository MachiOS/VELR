import sys
from math import log
import pandas as pd
import os
from tqdm import tqdm

import numpy as np


MIN = 10000

'''
logits_data_m1
logits_data_m2
logits_data_m3


for df in (logits_data_m1,logits_data_m2, )
logit = logits_data_m1[i][j]
min = logits_data_m1[i][0]
if logit < min:
    min = logit



'''


args = sys.argv

directory = args[1]

step = args[2]

num_model = int(args[3])
print(directory)


print(step)


# Exaple;
#
#   python get_minprob_mnist.py /media/iec/Seagate\ Expansion\ Drive/mnist_classifier 5000 100
#   python get_minprob_mnist.py /media/iec/Seagate\ Expansion\ Drive/cifar_classifier 4000 100
#   python get_minprob_mnist.py /media/iec/Seagate\ Expansion\ Drive/cifar_classifier 9000 100


def find_minProb(whole_data, output_prob_file, output_prob_kth_file, num_class=10):

    index_columns = ['index'] + [i for i in range(num_class)]
    columns = [i for i in range(num_class)]

    min_prob_df = pd.DataFrame(columns=index_columns)
    min_prob_kth_df = pd.DataFrame(columns=index_columns)

    # print(len(model_df_list[0]))

    num_data = whole_data.shape[0]

    min_probs = np.amin(whole_data, axis=2)

    # print(min_probs.shape) # (num_data, num_class)
    # print(min_probs[:3])

    min_prob_df = pd.DataFrame(min_probs, columns=columns)

    index_data = [[i] for i in range(num_data)]
    index_df = pd.DataFrame(index_data, columns=['index'])

    min_prob_df = index_df.join(min_prob_df, how='right')

    # previous_id = id

    min_prob_df.to_csv(output_prob_file, index=False)
    # min_prob_kth_df.to_csv(output_prob_kth_file,index=False)


def main():

    num_class = 100
    # num_class = 100 #CIFAR-100

    #  ============== FOR TEST DATA (start)==========================================================================
    for i in range(0, num_model):

        # prob_file = '/media/iec/Seagate Expansion Drive/mnist_classifier/model' + str(i) + '/test_prob'+ str(step) +'.csv'

        # if i < 200:
        #     sub_folder = '/M_100'
        # elif 200 <=  i  and i < 300:
        #     sub_folder = '/M_200'
        # elif 300 <=  i  and i < 500:
        #     sub_folder = '/M_300'
        # elif 500 <=  i  and i < 600:
        #     sub_folder = '/M_500'
        # else:
        #     sub_folder = '/M_600'

        sub_folder = '/M_0'

        prob_file = directory + sub_folder + '/model' + \
            str(i) + '/test_prob' + str(step) + '.csv'

        print(prob_file)

        try:
            data = pd.read_csv(prob_file, skiprows=1, header=None)
            columns_list = ['index'] + [i for i in range(num_class)]
            data.columns = columns_list

            data = data.drop(['index'], axis=1)

            if i == 0:
                whole_data = data.to_numpy()
                whole_data = np.expand_dims(whole_data, axis=1)
                # print(whole_data.shape)
            else:

                data_numpy = data.to_numpy()
                data_numpy = np.expand_dims(data_numpy, axis=1)
                # data_numpy.shape --> (num_data, 1, num_class+1)

                whole_data = np.concatenate((whole_data, data_numpy), axis=1)

                # print(data_numpy.shape)

                # print(whole_data.shape)

        except Exception as e:
            print(e)
            print('no file exists at {0}'.format(i))

    # print(whole_data.shape)

    whole_data = np.transpose(whole_data, (0, 2, 1))

    folder_path = directory + '/min_prob_normal'

    # folder_path = directory + '/step400_startidx_all/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    output_prob_file = folder_path + '/test_prob_min_' + str(step) + '.csv'
    output_prob_kth_file = folder_path + \
        '/test_prob_kth_model_min_' + str(step) + '.csv'

    print(output_prob_file)

    find_minProb(whole_data, output_prob_file, output_prob_kth_file, num_class)


if __name__ == '__main__':
    main()
