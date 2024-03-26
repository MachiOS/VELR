
import sys
from random import sample
import pandas as pd
import os


import numpy as np
import numpy.random as rd
import scipy as sp
import scipy.stats as st


from sklearn import mixture


from tqdm import tqdm

from multiprocessing import Pool


MIN = 10000


args = sys.argv

directory = args[1]

step = args[2]

num_model = int(args[3])
print(directory)


print(step)

'''
python get_minprob_GMM_v2.py /media/iec/Seagate\ Expansion\ Drive/mnist_classifier 5000 100

python get_minprob_GMM_v2.py /media/iec/Seagate\ Expansion\ Drive/cifar_classifier/seed1001_data100 3000 130

python get_minprob_GMM_v2.py /media/iec/Seagate\ Expansion\ Drive/cifar_classifier 9000 900

python get_minprob_GMM_v2.py /media/iec/Seagate\ Expansion\ Drive/cifar_classifier 9000 200
'''


def find_minProb(whole_data, output_dir,  num_class=10, ):

    output_prob_file = output_dir + '/test_prob_min_' + str(step) + '.csv'
    # output_prob_kth_file = output_dir + '/test_prob_kth_model_min_' + str(step) + '.csv'
    output_mean_file = output_dir + '/test_mean_' + str(step) + '.csv'
    output_variance_file = output_dir + '/test_variance_' + str(step) + '.csv'

    index_columns = ['index'] + [i for i in range(num_class)]

    min_prob_df = pd.DataFrame(columns=index_columns)
    # min_prob_kth_df = pd.DataFrame(columns=index_columns)
    output_mean_df = pd.DataFrame(
        columns=['index', 'label', 'majority', '2nd'])
    output_variance_df = pd.DataFrame(
        columns=['index', 'label', 'majority', '2nd'])

    # print(len(model_df_list[0]))

    mean_list = []
    variance_list = []
    majority_class_list = []

    num_data = whole_data.shape[0]
    print(num_data)
    # num_data = 3

    print(whole_data.shape)

    mean_list = []
    variance_list = []
    majority_class_list = []

    for i in tqdm(range(num_data)):
        sample_data = whole_data[i]

        mean_list_per_sample = []
        variance_list_per_sample = []
        majority_class_per_sample = []
        min_prob_list_per_sample = []

        for j in range(num_class+1):

            if j == 0:
                id = sample_data[j][0]
                min_prob_list_per_sample.append(id)
            else:
                prob_array = np.asarray(sample_data[j]).reshape(-1, 1)
                mgg = mixture.GaussianMixture(
                    n_components=2, covariance_type="full")

                model = mgg.fit(prob_array)

                predicted = model.fit_predict(prob_array)

                mean = model.means_
                variance = model.covariances_
                log_weights = np.exp(model._estimate_log_weights())
                majority_cluster = np.argmax(log_weights)

                new_data = prob_array[predicted == majority_cluster]
                min_in_majority = np.min(new_data)

                mean_list_per_sample.append(mean.tolist())
                # shape(mean.tolist()) = (num_cluster,1)

                variance_list_per_sample.append(variance.tolist())
                # shape(variance.tolist()) = (num_cluster,1, 1)

                majority_class_per_sample.append(majority_cluster)

                min_prob_list_per_sample.append(min_in_majority)

        min_prob_df.loc[i] = min_prob_list_per_sample
        # min_prob_kth_df.loc[i] = min_prob_kth_list

        mean_list.append(mean_list_per_sample)
        # shape = (num_data, num_class, num_cluster, 1)
        variance_list.append(variance_list_per_sample)
        # shape = (num_data, num_class, num_cluster, 1, 1)
        majority_class_list.append(majority_class_per_sample)

    # ====LOOP FOR ALL SAMPELS ENDS HERE======================================================

    # +++++ output estimated means +++++++++++++++++
    data_count = -1
    for index, mean_all in enumerate(mean_list):
        for label, means in enumerate(mean_all):
            data_count += 1
            final_mean = []
            for cluster, mean in enumerate(means):
                if cluster == majority_class_list[index][label]:

                    final_mean.insert(0, next(iter(mean)))
                else:
                    final_mean.append(next(iter(mean)))

            output_mean_data = [index, label] + final_mean
            output_mean_df.loc[data_count] = output_mean_data

    # ++++ output estimated variances +++++++++++++
    data_count = -1
    for index, variance_all in enumerate(variance_list):
        for label, variances in enumerate(variance_all):
            data_count += 1
            final_variance = []
            for cluster, variance in enumerate(variances):
                if cluster == majority_class_list[index][label]:

                    final_variance.insert(0, variance[-1][-1])
                else:
                    final_variance.append(variance[-1][-1])

            output_variance_data = [index, label] + final_variance
            output_variance_df.loc[data_count] = output_variance_data

    min_prob_df.to_csv(output_prob_file, index=False)
    output_mean_df.to_csv(output_mean_file, index=False)
    output_variance_df.to_csv(output_variance_file, index=False)


def main():

    num_class = 10
    # num_class = 100 for CIFAR-100

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

        sub_folder = '/M_0'

        prob_file = directory + sub_folder + '/model' + \
            str(i) + '/test_prob' + str(step) + '.csv'
        # prob_file = directory +'/model' + str(i) +  '/test_prob'+ str(step) +'.csv'

        print(prob_file)

        try:
            data = pd.read_csv(prob_file, skiprows=1, header=None)
            columns_list = ['index'] + [i for i in range(num_class)]
            data.columns = columns_list

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

    # print(whole_data[:2][:][:])

    # whole_data.shape before transpose --> (num_data, num_models, num_class+1)
    whole_data = np.transpose(whole_data, (0, 2, 1))
    # whole_data.shape after transpose  --> (num_data, num_class+1, num_models)

    # print(whole_data.shape)
    # print(whole_data[:2][:][:])

    folder_path = directory + '/min_prob'

    # folder_path = directory + '/step400_startidx_all/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    find_minProb(whole_data, folder_path, num_class)


if __name__ == '__main__':
    main()
