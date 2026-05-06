import mne
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from mne.decoding import SlidingEstimator, cross_val_multiscore
from scipy.stats import ttest_1samp
from pathlib import Path


class Participant:

    def __init__(self, subject, base):
        self.subject = subject
        self.syn_path = f"{base}/s{subject}/syn{subject}-s253-clean.set"
        self.non_syn_path = f"{base}/s{subject}/syn{subject}-s254-clean.set"
        self.matrix_syn = None
        self.matrix_nonsyn = None
        self.syn_max_trials = None
        self.non_syn_max_trials = None

        self.sub_accuracy = None
        self.time_point_accuracy = []
        self.max_accuracy_window = None
        self.N400_window = None
        self.result_txt_line = [f"Report For Subject{subject}"]

    def load_data(self):
        ### one window slice is about 5 ms data
        self.epochs_syn = mne.read_epochs_eeglab(self.syn_path)
        print(f"self epoch syn data shape: {self.epochs_syn.get_data().shape}")
        self.epochs_nonsyn = mne.read_epochs_eeglab(self.non_syn_path)
        self.epochs_syn.crop(tmin=-0.2, tmax=1)
        self.epochs_nonsyn.crop(tmin=-0.2, tmax=1)
        self.epochs_syn.apply_baseline(baseline=(-0.2, 0.0))
        self.epochs_nonsyn.apply_baseline(baseline=(-0.2, 0.0))
        self.syn_max_trials = self.epochs_syn.get_data().shape[0]
        self.non_syn_max_trials = self.epochs_nonsyn.get_data().shape[0]
        self.epochtime = self.epochs_syn.get_data().shape[2]

    def equal_matrix_svm(self, n_trials=5):
        min_trials = min(self.syn_max_trials, self.non_syn_max_trials)
        self.trials_for_pesudo = min_trials // n_trials
        trials_to_keep = self.trials_for_pesudo * n_trials

        syn_data = self.epochs_syn.get_data()
        nonsyn_data = self.epochs_nonsyn.get_data()

        syn_slice = np.random.permutation(syn_data.shape[0])[:trials_to_keep]
        nonsyn_slice = np.random.permutation(nonsyn_data.shape[0])[:trials_to_keep]

        self.matrix_syn = syn_data[syn_slice]
        self.matrix_nonsyn = nonsyn_data[nonsyn_slice]

    def create_pseudo_trials(self, matrix_1, matrix_2, n_trials=5):
        # get matrixs from conditions and create pseudo-trials by collapsing across n_trials
        pseudo_syn_list = []
        pseudo_nonsyn_list = []

        for time in range(self.trials_for_pesudo):
            start_trial = time * n_trials
            end_trial = start_trial + n_trials
            syn_group = matrix_1[start_trial:end_trial]
            nonsyn_group = matrix_2[start_trial:end_trial]
            average_syn = syn_group.mean(axis=0)
            average_nonsyn = nonsyn_group.mean(axis=0)
            pseudo_syn_list.append(average_syn)
            pseudo_nonsyn_list.append(average_nonsyn)
            start_trial += n_trials
            end_trial += n_trials
        pseudo_syn_array = np.array(pseudo_syn_list)
        pseudo_nonsyn_array = np.array(pseudo_nonsyn_list)

        return pseudo_syn_array, pseudo_nonsyn_array

    def make_svm_matrix(self):
        ### flattern to row by electrode and collapse all time and trial to columns
        syn_pseudo, nonsyn_pseudo = self.create_pseudo_trials(
            self.matrix_syn, self.matrix_nonsyn
        )
        self.x = np.concatenate((syn_pseudo, nonsyn_pseudo), axis=0)
        syn_label = np.zeros(len(syn_pseudo))
        nonsyn_label = np.ones(len(nonsyn_pseudo))
        self.y = np.concatenate((syn_label, nonsyn_label), axis=0)

    def run_svm(self):
        ### flattern the columns with time and trial() take time as a feature,
        ### re-organize the matrix to make a large long matrix columns by trial, timeframe, electrodes....
        ### logistic classifier, try random forest model
        pipeline = make_pipeline(StandardScaler(), SVC(kernel="linear"))
        time_step = SlidingEstimator(pipeline, n_jobs=1, scoring="accuracy")
        scores_across_time_matrix = cross_val_multiscore(
            time_step, self.x, self.y, cv=5, n_jobs=1
        )
        return scores_across_time_matrix

    def draw_lineplot_by_time(self, x_min=-200, x_max=800):
        collapsed_result = np.vstack(self.iteration_array)

        self.mean_score = collapsed_result.mean(axis=0)

        self.time = self.epochs_syn.times * 1000
        chance_level = 0.5

        plt.plot(
            self.time,
            self.mean_score,
            color="green",
            label="SVM Decoding Scores",
        )
        plt.axhline(
            chance_level, color="grey", linestyle="--", label="random chance (.5)"
        )
        plt.axvline(0, color="grey", linestyle="-", label="stimulus onset")
        plt.fill_between(
            self.time,
            self.mean_score,
            0.5,
            where=(self.mean_score > 0.5),
            color="lightgrey",
        )
        plt.xlim(x_min, x_max)
        plt.xticks(np.arange(x_min, x_max + 1, 100))

        plt.title(f"SVM Decoding Line Plot For Subject_{self.subject}")
        plt.xlabel("Time(ms)")
        plt.ylabel("Decoding Score (Accuracy)")

        plt.legend(loc="lower right")
        plt.savefig(f"./figures/{self.subject}_lineplot.png")
        plt.close()

    def run_statistics(self, window_start=0, window_end=800, bin_size=100, alpha=0.05):
        self.result_txt_line.append("========== Statistical result ==========")
        self.result_txt_line.append(
            f"The pseudo_trial_matrix for SVM is {self.x.shape}, labels for condition is {self.y.shape}"
        )
        data = np.vstack(self.iteration_array)
        bins = np.arange(window_start, window_end + 1, bin_size)

        for i in range(len(bins) - 1):
            start, end = bins[i], bins[i + 1]
            time_window = (self.time >= start) & (self.time < end)
            window_accuracies = data[:, time_window].mean(axis=1)

            t_stat, p_val_two_tailed = ttest_1samp(window_accuracies, 0.5)
            if t_stat > 0:
                p_value_one_tailed = p_val_two_tailed / 2
                significance = p_val_two_tailed < alpha
            else:
                p_value_one_tailed = 1
                significance = False

            self.result_txt_line.append(
                f"Window {start}-{end} ms"
                f"Mean_Accuracy: {(np.mean(window_accuracies)):.3f},"
                f"t-value = {t_stat:.3f},"
                f"p-value = {p_value_one_tailed:.3f},"
                f"significance: {significance}"
            )

    def write_participant_result(self):
        filename = f"./output/SVMresult_{self.subject}.txt"
        with open(filename, "w") as fh:
            fh.write("\n".join(self.result_txt_line))


def group_stat_analysis(
    matrix, window_start=-200, window_end=800, bin_size=100, alpha=0.05
):
    group_result = []
    group_result.append("=====Group Statistics=====")
    group_result.append("Number of Participants: {matrix[0]}")

    time_axis = matrix.shape[1]
    time_value = np.linspace(window_start, window_end + 1, time_axis)
    # cut the axis of -200 to 1000 into 241 slices
    bins = np.arange(window_start, window_end + bin_size, bin_size)
    for i in range(len(bins) - 1):
        start, end = bins[i], bins[i + 1]
        time_window = (time_value >= start) & (time_value < end)
        col_slice = np.where(time_window)[0]
        window_accuracies = matrix[:, col_slice].mean(axis=1)
        group_result.append(f"Accuracy From Each Participant = {window_accuracies}")
        t_stat, p_val_two_tailed = ttest_1samp(window_accuracies, 0.5)
        if t_stat > 0:
            p_value_one_tailed = p_val_two_tailed / 2
            significance = p_val_two_tailed < alpha
        else:
            p_value_one_tailed = 1
            significance = False
        group_result.append(
            f"Window {start}-{end} ms"
            f"Mean_Accuracy: {(np.mean(window_accuracies)):.3f},"
            f"t-value = {t_stat:.3f},"
            f"p-value = {p_value_one_tailed:.3f},"
            f"significance: {significance}",
        )
    filename = f"./output/SVMresult_group_stat.txt"
    with open(filename, "w") as fh:
        fh.write("\n".join(group_result))


def draw_lineplot_by_time_group(matrix, x_min=-200, x_max=800):

    group_mean_score = matrix.mean(axis=0)

    time_axis = matrix.shape[1]
    time_value = np.linspace(x_min, x_max + 1, time_axis)

    chance = 0.5
    plt.plot(
        time_value,
        group_mean_score,
        color="green",
        label="SVM Decoding Scores",
    )
    plt.axhline(chance, color="grey", linestyle="--", label="random chance (.5)")
    plt.axvline(0, color="grey", linestyle="-", label="stimulus onset")
    plt.fill_between(
        time_value,
        group_mean_score,
        0.5,
        where=(group_mean_score > 0.5),
        color="lightgrey",
    )
    plt.xlim(x_min, x_max)
    plt.xticks(np.arange(x_min, x_max + 1, 100))

    plt.title("SVM Decoding Line Plot For Group-Level")
    plt.xlabel("Time(ms)")
    plt.ylabel("Decoding Score (Accuracy)")

    plt.legend(loc="lower right")
    plt.savefig(f"./figures/group_level_lineplot.png")
    plt.close()


def main():
    participant_dir = Path("./data")
    group_list = []
    for folder in participant_dir.iterdir():
        if folder.name.startswith(".") or not folder.is_dir():
            continue
        subject_num = folder.name.replace("s", "")
        print(f"subject number is {subject_num}")
        folder = Participant(subject_num, base="./data")
        folder.load_data()
        iteration_result = []
        iteration_times = 1
        for i in range(iteration_times):
            print(f"Iteration for {i+1}/{iteration_times} time")
            folder.equal_matrix_svm()
            folder.make_svm_matrix()
            single_result = folder.run_svm()
            iteration_result.append(single_result)

        folder.iteration_array = np.array(iteration_result)
        folder.draw_lineplot_by_time()
        folder.run_statistics()
        folder.write_participant_result()
        group_list.append(folder.mean_score)
    group_matrix = np.vstack(group_list)
    group_stat_analysis(group_matrix)
    draw_lineplot_by_time_group(group_matrix)


if __name__ == "__main__":
    main()
