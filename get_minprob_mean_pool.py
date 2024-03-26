

import sys
import pandas as pd
import os


import numpy as np


from tqdm import tqdm

from multiprocessing import Pool

import shutil


MIN = 10000


args = sys.argv

directory = args[1]

step = args[2]

num_model = int(args[3])

print(directory)


print(step)

'''

python get_minprob_mean_pool.py /media/iec/Seagate\ Expansion\ Drive/cifar_classifier/seed1001_data1500 9000 100


python get_minprob_mean_pool.py /media/iec/Expansion18/cifar_classifier/seed2002_data1500 9000 100
'''


def find_meanProb(whole_data, output_dir, num_class=10):

    output_mean_file = output_dir + '/test_prob_mean_all_' + str(step) + '.csv'

    index_columns = ['index'] + [i for i in range(num_class)]
    columns = [i for i in range(num_class)]

    mean_prob_df = pd.DataFrame(columns=index_columns)

    # print(len(model_df_list[0]))

    num_data = whole_data.shape[0]

    mean_probs = np.mean(whole_data[:, 1:, :], axis=2)

    # print(mean_probs.shape) # (num_data, num_class)
    # print(mean_probs[1])

    # print(mean_probs[:,1:].shape)
    mean_prob_df = pd.DataFrame(mean_probs, columns=columns)

    index_data = [[i] for i in range(num_data)]
    index_df = pd.DataFrame(index_data, columns=['index'])

    mean_prob_df = index_df.join(mean_prob_df, how='right')

    # previous_id = id

    mean_prob_df.to_csv(output_mean_file, index=False)
    # min_prob_kth_df.to_csv(output_prob_kth_file,index=False)


def read_csv(filename):
    'converts a filename to a pandas dataframe'
    num_class = 100
    # columns_list = ['index'] + [i for i in range(num_class)]
    data = pd.read_csv(filename, skiprows=1, header=None)
    # data.columns= columns_list

    data_numpy = data.to_numpy()
    data_numpy = np.expand_dims(data_numpy, axis=1)
    # print(data_numpy.shape)
    # data_numpy.shape --> (num_data, 1, num_class+1)

    return data_numpy


def main():

    num_class = 100
    # num_class = 100 for CIFAR-100

    file_list = []
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
        # elif 600 <=i and i < 1000:
        #     sub_folder = '/M_600'
        # elif 1000 <=  i  and i < 1400:
        #     sub_folder = '/M_1000'
        # elif 1400 <=  i  and i < 2000:
        #     sub_folder = '/M_1400'
        # elif 2000 <=  i  and i < 3000:
        #     sub_folder = '/M_2000'
        # elif 3000 <=  i  and i < 4000:
        #     sub_folder = '/M_3000'
        # elif 4000 <=  i  and i < 5000:
        #     sub_folder = '/M_4000'

        # if i < 1000:
        #     sub_folder = '/M_0'
        # elif 1000 <=  i  and i < 2000:
        #     sub_folder = '/M_1000'
        # elif 2000 <=  i  and i < 3000:
        #     sub_folder = '/M_2000'
        # elif 3000 <=  i  and i < 4000:
        #     sub_folder = '/M_3000'
        # elif 4000 <=  i  and i < 5000:
        #     sub_folder = '/M_4000'

        sub_folder = '/M_0'

        prob_file = directory + sub_folder + '/model' + \
            str(i) + '/test_prob' + str(step) + '.csv'
        # prob_file = directory +'/model' + str(i) +  '/test_prob'+ str(step) +'.csv'

        # print(prob_file)

        file_list.append(prob_file)

    print('number of csv files', len(file_list))

    with Pool(processes=8) as pool:
        data_numpy_list = pool.map(read_csv, file_list)
        print(len(data_numpy_list))

        whole_data = np.concatenate(data_numpy_list, axis=1)

    whole_data = np.transpose(whole_data, (0, 2, 1))

    print(whole_data.shape)

    # folder_path = directory + '/mean_prob'
    folder_path = directory + '/results_0_' + str(num_model) + '/mean_prob'

    # folder_path = directory + '/step400_startidx_all/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    find_meanProb(whole_data, folder_path, num_class)

    # ***** let this be M_0
    label_file = directory + f'/M_0/model0/test_label{step}.csv'
    label_file_copy = folder_path + f'/test_label{step}.csv'
    shutil.copyfile(label_file, label_file_copy)


if __name__ == '__main__':
    main()
