# BCOG200_Final_Project [SVM on ERP data (data source: open-source cleaned N400 data)]
# paper of reference: DOI: 10.1111/psyp.14570


# The planned project is to create a pipeline for running SVM on ERP data; The project comes from a research question of 'can machine learning methods define a more precise time window for an Event-Related Potential component Nref, whose temporal boundaries are not clearly defined'. current literature studied Nref within a very broad window of 400-1200ms. The goal of this project is to replicate the N400 from Luck's paper on python, and the same model will be used with my first year project of NRef signal to cross-validate the Nref effect



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

    def load_data(self): #this method load the participant ERP data
    def equal_matrix_svm(self): #this method find the matrix size equating conditions for training model
    def create_pseudo_trials(self, matrix_1, matrix_2): #this method create pseudo trials from the data (collapsing trials)
    def make_svm_matrix(self): #this method combine the condition matrix and create answer key column
    def run_svm(full_matrix, y_col): #this method is running the model and return accuracy at each time point
	def draw_accuracy_plot(self): #this method is comparing the accuracy across time point, plot the accuracy and time point
	def average_accuracy(trial_accuracy): #this method average accuracy of an interval e.g. collapse accuracy for every 100ms and compare the window accuracy
	

	
	
