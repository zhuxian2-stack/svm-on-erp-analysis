import mne


# file_path = "/Users/kathleenz/Desktop/UIUC/Coursework/BCOG200/final_project/STUDY5subjects/s02/syn02-s253-clean.set"

# epochs = mne.read_epochs_eeglab(file_path)


# # 2. Print a general summary of the dataset
# print(epochs)

# # 3. Inspect the time information specifically
# print(f"Time window: {epochs.tmin} seconds to {epochs.tmax} seconds")
# print(f"Sampling frequency: {epochs.info['sfreq']} Hz")
# print(f"Total number of time points per epoch: {len(epochs.times)}")

# # 4. Look at the actual array of time points (in seconds)
# # This shows you the exact millisecond steps (e.g., -0.200, -0.196, -0.192...)
# print("\nFirst 10 time points:")
# print(epochs.times[:10])


# epochs_syn = mne.read_epochs_eeglab("./STUDY5subjects/s02/syn02-s253-clean.set")
# epochs_syn.crop(tmin=-0.2, tmax=0.8)
# # epochs_syn.apply_baseline(baseline=(-0.2, 0.0))

# # data_syn = epochs_syn.get_data()
# # print(f"Synonymous Matrix Shape: {data_syn.shape}")

# num_trials = len(epochs_syn)

# print(f"I have exactly {num_trials} trials!")
# print(epochs_syn.times)
# print(epochs_syn.get_data().shape)
import numpy as np


class Participant:

    def __init__(self, subject, base):
        self.subject = subject
        self.syn_path = f"{base}/{subject}/syn.set"
        self.non_syn_path = f"{base}/{subject}/non-syn.set"
        self.matrix_syn = None
        self.matrix_nonsyn = None
        self.syn_max_trials = None
        self.non_syn_max_trials = None

        self.sub_accuracy = None
        self.time_point_accuracy = []
        self.max_accuracy_window = None
        self.N400_window = None

    def load_data(self):
        self.epochs_syn = mne.read_epochs_eeglab(self.syn_path)
        # print(len(self.epochs_syn))
        # print(self.epochs_syn.get_data().shape)
        print(self.epochs_syn.event_id)
        self.epochs_nonsyn = mne.read_epochs_eeglab(self.non_syn_path)
        self.epochs_syn.apply_baseline(baseline=(-0.2, 0.0))
        self.epochs_nonsyn.apply_baseline(baseline=(-0.2, 0.0))
        self.syn_max_trials = self.epochs_syn.get_data().shape[0]
        self.non_syn_max_trials = self.epochs_nonsyn.get_data().shape[0]
        self.epochtime = self.epochs_syn.get_data().shape[2]

    #### here introduce the label column
    #### double check if the data is trial-level or collapsed data
    def equal_matrix_svm(self):
        min_trials = min(self.syn_max_trials, self.non_syn_max_trials)

        self.trials_for_pesudo = min_trials // 5

        self.matrix_syn = np.random.permutation(self.epochs_syn.get_data())[:min_trials]

        self.matrix_nonsyn = np.random.permutation(self.epochs_nonsyn.get_data())[
            :min_trials
        ]
        ###print out sth like "max trials for equating the matrix is xxx "

    def create_pseudo_trials(self, matrix_1, matrix_2):
        pseudo_syn_list = []
        pseudo_nonsyn_list = []

        # random_syn_matrix = np.random.permutation(matrix_1)
        # print(random_syn_matrix.shape)
        # random_nonsyn_matrix = np.random.permutation(matrix_2)
        start_trial = 0
        end_trial = 5
        for time in range(self.trials_for_pesudo):
            syn_group = matrix_1[start_trial:end_trial]
            nonsyn_group = matrix_2[start_trial:end_trial]
            average_syn = syn_group.mean(axis=0)
            average_nonsyn = nonsyn_group.mean(axis=0)
            pseudo_syn_list.append(average_syn)
            pseudo_nonsyn_list.append(average_nonsyn)
            start_trial += 5
            end_trial += 5
        pseudo_syn_array = np.array(pseudo_syn_list)
        pseudo_nonsyn_array = np.array(pseudo_nonsyn_list)
        print(pseudo_syn_array.shape)

        return pseudo_syn_array, pseudo_nonsyn_array

    def make_svm_matrix(self):
        syn_pseudo, nonsyn_pseudo = self.create_pseudo_trials(
            self.matrix_syn, self.matrix_nonsyn
        )
        self.x = np.concatenate((syn_pseudo, nonsyn_pseudo), axis=0)
        syn_label = np.zeros(len(syn_pseudo))
        nonsyn_label = np.ones(len(nonsyn_pseudo))
        self.y = np.concatenate((syn_label, nonsyn_label), axis=0)
        # print(self.y.shape)
        print(
            f"The matrix for SVM is {self.x.shape}, labels for condition is {self.y.shape}"
        )

    #### def decoding_accuracy_graph_in_ms(self)
    #### do I need to create a matrix with what group the trials are in for accuracy confirmation


def main():
    participant_list = ["s02"]
    # , "s05", "s07", "s08", "s10"]
    for n in participant_list:
        n = Participant(n, base="./STUDY5subjects")
        # n.load_data()
        # n.equal_matrix_svm()
        # # syn_pseudo, nonsyn_pseudo = n.create_pseudo_trials(n.matrix_syn, n.matrix_nonsyn)
        # n.make_svm_matrix()


if __name__ == "__main__":
    main()
