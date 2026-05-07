import mne
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from mne.decoding import SlidingEstimator, cross_val_multiscore
from scipy.stats import ttest_1samp
from pathlib import Path
from scipy.ndimage import gaussian_filter1d


def draw_lineplot_by_time(
    x: np.ndarray | int,  # usually time axis or number of columns in time
    y: np.ndarray,  # usually mean scores
    subject_id: str,
    x_min: float = -200,
    x_max: float = 800,
    line_color: str = "green",
    fill_between_color: str = "lightgrey",
    figure_title_prefix: str = "SVM Decoding Line Plot For",  # make sure to replace
    figure_title_type: str = "Subject",
    line_label: str = "SVM Decoding Scores",
    xlabel: str = "Time (ms)",
    ylabel: str = "Decoding Score (Accuracy)",
    xtick_step_size: float = 100,
    chance_level: float = 0.5,
    legend_location: str = "lower right",
    save_figure: bool = True,
    figure_output_path: Path | str = Path("./figures"),
    figure_output_extension: str = ".png",
):
    # TODO: move this logic outside to get your proper input arrays

    if isinstance(x, int):  # this is for updating data structure for group-level plot
        x = np.linspace(x_min, x_max + 1, x)
    y_smoothed = gaussian_filter1d(y, sigma=1)
    plt.plot(
        x,
        y_smoothed,
        color=line_color,
        label=line_label,
    )

    # held in common across graphs
    plt.axhline(chance_level, color="grey", linestyle="--", label="random chance (.5)")
    plt.axvline(0, color="grey", linestyle="-", label="stimulus onset")

    plt.fill_between(
        x,
        y_smoothed,
        chance_level,
        where=(y_smoothed > chance_level),
        color=fill_between_color,
    )
    plt.xlim(x_min, x_max)
    plt.xticks(np.arange(x_min, x_max + 1, xtick_step_size))

    figure_title = f"{figure_title_prefix} {figure_title_type} {subject_id}"
    plt.title(figure_title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.legend(loc=legend_location)

    if save_figure:
        figure_filename = (
            figure_output_path / f"{subject_id}_lineplot{figure_output_extension}"
        )
        plt.savefig(figure_filename)

    plt.close()


def run_statistics(
    subject_id: str,
    txt_title: str,
    data: np.ndarray,
    time_axis: np.ndarray | int,  # time column for blocking time window of 100ms
    txt_report_list: list = [],
    window_start=0,
    window_end=800,
    bin_size=100,
    alpha=0.05,
):

    if isinstance(time_axis, int):
        time_axis = np.linspace(window_start, window_end + 1, time_axis)
    txt_report_list.append(txt_title)
    bins = np.arange(window_start, window_end + 1, bin_size)

    for i in range(len(bins) - 1):
        start, end = bins[i], bins[i + 1]
        time_window = (time_axis >= start) & (time_axis < end)
        col_slice = np.where(time_window)[0]
        window_accuracies = data[:, col_slice].mean(axis=1)

        t_stat, p_val_two_tailed = ttest_1samp(window_accuracies, 0.5)
        if t_stat > 0:
            p_value_one_tailed = p_val_two_tailed / 2
            significance = p_val_two_tailed < alpha
        else:
            p_value_one_tailed = 1
            significance = False

        txt_report_list.append(
            f"Window {start}-{end} ms"
            f"Mean_Accuracy: {(np.mean(window_accuracies)):.3f},"
            f"t-value = {t_stat:.3f},"
            f"p-value = {p_value_one_tailed:.3f},"
            f"significance: {significance}"
        )
        write_participant_result(txt_report_list, subject_id=subject_id)


def write_participant_result(
    text_list: list,
    subject_id: str,
    file_path_base=Path("./output"),
    file_path_extension: str = ".txt",
):
    file_name = file_path_base / f"SVMresult_{subject_id}{file_path_extension}"
    with open(file_name, "w") as fh:
        fh.write("\n".join(text_list))


class Participant:

    def __init__(self, subject: str, base: str | Path) -> None:
        self.subject = subject
        # Path way
        base = Path(base)
        self.syn_path = base / f"s{subject}" / f"syn{subject}-s253-clean.set"
        self.non_syn_path = base / f"s{subject}" / f"syn{subject}-s254-clean.set"
        self.matrix_syn: np.ndarray | None = None
        self.matrix_nonsyn: np.ndarray | None = None
        self.syn_max_trials: int | None = None
        self.non_syn_max_trials: int | None = None

    def load_data(self, pre_time: float = -0.2, post_time: float = 1) -> None:
        """load the participant eeglab data and transform by mne to python matrix"""
        self.epochs_syn: np.ndarray = mne.read_epochs_eeglab(self.syn_path)
        self.epochs_nonsyn: np.ndarray = mne.read_epochs_eeglab(self.non_syn_path)
        self.epochs_syn.crop(
            tmin=pre_time, tmax=post_time
        )  # cut the matrix into timw window of interest
        self.epochs_nonsyn.crop(tmin=pre_time, tmax=post_time)
        self.epochs_syn.apply_baseline(baseline=(pre_time, 0.0))
        self.epochs_nonsyn.apply_baseline(baseline=(pre_time, 0.0))
        self.syn_max_trials: int = self.epochs_syn.get_data().shape[0]
        self.non_syn_max_trials: int = self.epochs_nonsyn.get_data().shape[0]

    def equal_matrix_svm(self, n_trials: int = 5) -> None:
        """this function trim the matrix to equal size,
        by fitting # of minimal trials from the two conditions"""

        min_trials = min(self.syn_max_trials, self.non_syn_max_trials)
        self.trials_for_pesudo = min_trials // n_trials
        trials_to_keep = self.trials_for_pesudo * n_trials

        syn_data = self.epochs_syn.get_data()
        nonsyn_data = self.epochs_nonsyn.get_data()

        syn_slice = np.random.permutation(syn_data.shape[0])[:trials_to_keep]
        nonsyn_slice = np.random.permutation(nonsyn_data.shape[0])[:trials_to_keep]

        self.matrix_syn = syn_data[syn_slice]
        self.matrix_nonsyn = nonsyn_data[nonsyn_slice]

    def create_pseudo_trials(self, matrix_1, matrix_2, n_trials: int = 5) -> np.ndarray:
        """this function create pseudo trials by collapsing across n_trials"""
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

    def make_svm_matrix(self) -> None:
        """this function put matrix of two conditiosn together and assign key column (condition label)"""
        syn_pseudo, nonsyn_pseudo = self.create_pseudo_trials(
            self.matrix_syn, self.matrix_nonsyn
        )
        self.x = np.concatenate((syn_pseudo, nonsyn_pseudo), axis=0)
        syn_label = np.zeros(len(syn_pseudo))
        nonsyn_label = np.ones(len(nonsyn_pseudo))
        self.y = np.concatenate((syn_label, nonsyn_label), axis=0)

    def run_svm(self) -> np.ndarray:
        """this function is the pipeline for running linear svm"""
        pipeline = make_pipeline(StandardScaler(), SVC(kernel="linear"))
        time_step = SlidingEstimator(pipeline, n_jobs=1, scoring="accuracy")
        scores_across_time_matrix = cross_val_multiscore(
            time_step, self.x, self.y, cv=5, n_jobs=1
        )
        return scores_across_time_matrix


def main():
    participant_dir = Path("./data")
    group_list = []
    for folder in participant_dir.iterdir():
        if folder.name.startswith(".") or not folder.is_dir():
            continue
        subject_num = folder.name.replace("s", "")
        folder = Participant(subject_num, base="./data")
        folder.load_data()
        iteration_result = []
        iteration_times = 10
        for i in range(iteration_times):
            print(f"Iteration for {i+1}/{iteration_times} time")
            folder.equal_matrix_svm()
            folder.make_svm_matrix()
            single_result = folder.run_svm()
            iteration_result.append(single_result)

        folder.iteration_array = np.array(iteration_result)
        matrix_collapsed = np.vstack(folder.iteration_array)
        y_mean_score = matrix_collapsed.mean(axis=0)
        x_time = folder.epochs_syn.times * 1000

        draw_lineplot_by_time(x=x_time, y=y_mean_score, subject_id=folder.subject)
        run_statistics(
            data=matrix_collapsed,
            time_axis=x_time,
            subject_id=folder.subject,
            txt_report_list=[],
            txt_title=f"=====Participant {folder.subject} Statistical Report=====",
        )
        group_list.append(y_mean_score)
    group_matrix = np.vstack(group_list)
    group_accuracy = group_matrix.mean(axis=0)
    group_time = group_matrix.shape[1]
    draw_lineplot_by_time(x=group_time, y=group_accuracy, subject_id="Group-Level")
    run_statistics(
        data=group_matrix,
        time_axis=group_time,
        subject_id="Group-Level",
        txt_title=f"====={subject_num} Statistical Report=====",
        txt_report_list=[],
    )


if __name__ == "__main__":
    main()
