from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import argparse
import matplotlib as mpl
mpl.rcParams['axes.formatter.useoffset'] = False

parser = argparse.ArgumentParser(
    description='Get precision, recall, F1 scores for Uniform and GMM methods')

parser.add_argument('-save_dir', type=str,
                    default="results/mean_prob/test_label9000.csv",  help='model_end')

parser.add_argument('-gmm_file_path', type=str, default="min_prob/test_prob_max_9000.csv",
                    help='path of the GMM max prob file you want to get scores for')

parser.add_argument('-normal_file_path', type=str, default="min_prob/test_prob_max_9000.csv",
                    help='path of the normal max prob file you want to get scores for')

parser.add_argument('-uniform_file_path', type=str, default="results/mean_prob/test_prob_max_9000.csv",
                    help='path of the uniform max prob file you want to get scores for')

parser.add_argument('-start_model_num', type=int, default=0,
                    help='start model num')

parser.add_argument('-end_model_num', type=int, default=0,
                    help='end model num')

parser.add_argument('-num_classes', type=int, default=0,
                    help='end model num')

args = parser.parse_args()


def plot_f1_graphs(prob_df, f1_file_name, cf_file_name, save_dir, num_classes):
    prob_df.drop(["index"], axis=1, inplace=True)
    # print(prob_df)
    print("save", save_dir)

    pred = prob_df["max_cls"]
    correct = prob_df["label"].astype(int)

    cm = confusion_matrix(pred, correct)

    names = [str(i) for i in range(num_classes)]
    print(classification_report(correct, pred, target_names=names, digits=4))

    print("\nConfusion matrix:")
    print(cm)

    cm_df = pd.DataFrame(cm, index=names, columns=names)

    plt.figure(figsize=(20, 19))
    sns.heatmap(cm_df, annot=True)
    plt.title('Confusion Matrix')
    plt.ylabel('Actal Values')
    plt.xlabel('Predicted Values')
    # plt.show()
    plt.savefig(save_dir+"/"+cf_file_name)

    f1_vals = []
    thresholds = []
    rejection_rates = []
    # at_0_f1 = 0
    for (column_name, column_data) in prob_df.iteritems():
        if "thld" in column_name:
            # print(column_name)

            df2 = prob_df.loc[prob_df[column_name] == True]
            pred = df2["max_cls"]
            correct = df2["label"]
            # print(pred)
            # print(correct)
            f1_val = f1_score(correct, pred, average="macro")
            f1_vals.append(f1_val)
            thresholds.append(column_name.split("_")[1])

            rejection_df = prob_df.loc[prob_df[column_name] == False]
            # print(len(rejection_df["max_cls"]))
            rejection_rate = len(rejection_df["max_cls"]) / len(prob_df.index)
            rejection_rates.append(rejection_rate)

        # if column_name == "isCorrect":
        #     df2 = prob_df.loc[prob_df[column_name] == 1]
        #     pred = df2["max_cls"]
        #     correct = df2["label"]
        #     # print(pred)
        #     # print(correct)
        #     # at_0_f1 = f1_score(correct, pred, average="macro")
        #     # print("AT F1:", at_0_f1)
        #     thresholds.append(column_name.split("_")[1])

        #     rejection_df = prob_df.loc[prob_df[column_name] == False]
        #     # print(len(rejection_df["max_cls"]))
        #     rejection_rate = len(rejection_df["max_cls"]) / len(prob_df.index)
        #     rejection_rates.append(rejection_rate)

    print(rejection_rates)
    print(f1_vals)

    fig, ax = plt.subplots()
    # ax.yaxis.set_ticks(np.array([i/10 for i in range(11)]))
    ax.plot(thresholds, f1_vals, color="red", marker="o", label="F1 Score")
    ax.set_xlabel("Threshold", fontsize=10)
    ax.set_ylabel("F1 Score, Rejection Ratio", fontsize=10)
    ax.plot(thresholds, rejection_rates, color="blue",
            marker="o", label="Rejection Rates")

    plt.legend(loc="upper left")
    # twin object for two different y-axis on the sample plot
    # ax2 = ax.twinx()
    # ax2.plot(thresholds, rejection_rates, color="blue", marker="o")
    # ax2.set_ylabel("Rejection Rate", color="blue", fontsize=14)
    fig.savefig(save_dir+"/"+f1_file_name,
                format='jpeg',
                dpi=100,
                bbox_inches='tight')

    # plt.figure(figsize=(20, 19))
    plt.figure()
    plt.title('F1 vs 0 Threshold')
    plt.ylabel('Threshold')
    plt.xlabel('F1')
    # plt.show()
    plt.savefig(save_dir+"/"+"F1_0_threshold")

    # report
    # names = [str(i) for i in range(num_classes)]
    # print(classification_report(correct, pred, target_names=names, digits=4))

    # print("\nConfusion matrix:")
    # print(cm)

    # cm_df = pd.DataFrame(cm, index=names, columns=names)

    # plt.figure(figsize=(20, 19))
    # sns.heatmap(cm_df, annot=True)
    # plt.title('Confusion Matrix')
    # plt.ylabel('Actal Values')
    # plt.xlabel('Predicted Values')
    # # plt.show()
    # plt.savefig(save_dir+"/"+cf_file_name)


if __name__ == "__main__":
    save_dir = args.save_dir
    start_model_num = args.start_model_num
    end_model_num = args.end_model_num
    num_classes = int(args.num_classes)
    print("classes:", num_classes)

    # GMM:
    # inp_file_gmm = args.gmm_file_path
    # prob_df_gmm = pd.read_csv(inp_file_gmm)
    # plot_f1_graphs(
    #     prob_df_gmm, "GMM F1 and Rejection rate vs Threshold.jpg",
    #     "GMM_cf_test_prob_max.png", save_dir+"/min_prob", num_classes)

    # normal:
    inp_file_normal = args.normal_file_path
    prob_df_normal = pd.read_csv(inp_file_normal)
    plot_f1_graphs(prob_df_normal, "Normal F1 and Rejection rate vs Threshold.jpg",
                   "normal_cf_test_prob_max.png", save_dir+"/min_prob_normal", num_classes)

    # uniform:
    inp_file_uniform = args.uniform_file_path
    prob_df_uniform = pd.read_csv(inp_file_uniform)
    plot_f1_graphs(prob_df_uniform, "Uniform F1 and Rejection rate vs Threshold.jpg",
                   "uniform_cf_test_prob_max.png", save_dir+f"/results_{start_model_num}_{end_model_num}/mean_prob", num_classes)
