# BCOG200_Final_Project: Support Vector Machine on ERP data 


## Project Motivation:
The planned project is to create a pipeline for running SVM on ERP data; 
The project comes from a research question of 'can machine learning methods define a more precise time window for an Event-Related Potential component Nref, whose temporal boundaries are not clearly defined'. 
Current literature studied Nref within a very broad window of 400-1200ms.
The goal of this project is to replicate the N400 from Luck's paper on python, and the same model will be used with my first year project of NRef signal to cross-validate the Nref effect.

## Refined Abstract: 
Incremental referential processing is a key mechanism that allows immediate integration of relevant
context and linguistic information for identifying the intended referent. Instead of waiting for a sentence
to finish, humans actively try to figure out what the speaker or the writer is referring to while taking the
context into account. Referential ambiguity emerges when people fail to predict the referent based on the
information in the discourse. Nieuwland et al., (2007) found an ERP component (NRef) that seems to
reflect referential ambiguity (e.g.‘Jim talked to the nephew who is into…
’ with 2 nephews vs. 1 nephew present). However, instead of a clear window defined as other ERP components (e.g. N400), NRef is
defined as a sustained negativity after the stimulus onset. Literature studied Nref used various time
windows to indicate the Nref effect, from 600 to 1200ms. Carrasco et al (2024) investigated the method of
support vector machine (SVM) decoding analysis on ERP data, and the result showed a larger effect size
and similar time window with the traditional univariate approach. Referential ambiguity emerges from
context, therefore, the Nref effect might not only reflect the ambiguity processing between conditions, but
also confound with other processing such as memory retrieval (which nephew is available) and
conceptual similarity between referents (e.g.‘the nephew who is into history’ shared ‘the nephew’ with
‘the nephew who is into politics’). The decoding accuracy from SVM allows us to investigate whether
there is different processing and if they can be decoded. The current project is to construct the framework
for applying SVM to the Nref data. Nref data from my project is not yet available and there is no open
source Nref data found online, N400 data from Kappenman et al 2020 will be used as the example
actual training of the model (which regression to apply for the model), and plot making (also analysis for
statistical difference).
input. The code is currently done with the preparation of matrix for training, the next step will be the actual training of the model (which regression to apply for the model), and plot making (also analysis for
statistical difference)


## Detail Information:
### Input Data Format:
This project accepts ERPlab EEG data format: xxx.set. The data format should be: for each participant folder, it has a cleaned .set file for each condition of the experiment (e.g. condition1.set, condition2.set). 


### Example Use Cases: 
As mentioned in the motivation, the prospect for this code is to navigate a better window for a component (NRef) with an ambiguous time window of interest, which gives a higher validity comparing to navigate the window by looking at the data pattern. 


### Implementation Plan:
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
	def overall_accuracy(self): #this method is generating the accuracy comparison for conditions(collapsing the decoding accuracy for all participants) and return a graph.
	
	
### Planned Analyses
The plan is to decode the ERP data (two condition comparison) with Support Vector Machine, the outcome of the .py will be a SVM decoding accuracy graph at each time point for each participant, and a collective decoding accuracy for the population across time points. Also a .txt report of the result of the decoding.    


### Testing Section
The output of the code will be figures and a text file report of the decoding result.
1. participant level decoding accuracy line plot
2. Group level decoding accuracy line plot
3. report of individual level decoding, t-test result report, and report of time window of high decoding accuracy, also the group level decoding report. 

