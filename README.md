# BCOG200_Final_Project [SVM on ERP data (data source: open-source cleaned N400 data)]
# paper of reference: DOI: 10.1111/psyp.14570


# The planned project is to create a pipeline for running SVM on ERP data; The project comes from a research question of 'can machine learning methods define a more precise time window for an Event-Related Potential component Nref, whose temporal boundaries are not clearly defined'. current literature studied Nref within a very broad window of 400-1200ms. The goal of this project is to replicate the N400 from Luck's paper on python, and the same model will be used with my first year project of NRef signal to cross-validate the Nref effect



class Participant:
    def __init__(self, subj_id):
        self.subj_id=subj_id
        
        
        self.cond_1_trials=None
        self.cond_2_trials=None
        self.epoch_time=None
		
    	self.sub_accuracy=None
		self.time_point_accuracy=[]
		self.max_accuracy_window=None


	def load_data():
		pass
		#x,y=read data


def equate_trials(x,y):
	pass
	#min(x,y) this is for getting the equal num of trials for svm(go with the smaller trials across conditions)
	#Return x, y


def pseudo_trials(data, num_trials):
  #this creates pseudotrials for train and test
	Return matrix

def combine_matrix(matrix1, matrix2):
  #combine the matrix of two conditions
  return full_matrix, y_col


def run_svm(full_matrix, y_col):
	#svm run on every time point and get accuracy at each time point
	return snapshot_accuracy

def time_window_compare(interval):
	#Cut the time_point_accuracy into n interval, get the average accuracy across diff time window e.g. 100ms, 200ms...
	#Compare accuracy across time window
	return time_window_of_highest_accuracy


def average_accuracy(time_point_accuracy):
	pass
	#Average across all time points


def group_accuracy(sub_accuracy):
	pass
	#Average across group
