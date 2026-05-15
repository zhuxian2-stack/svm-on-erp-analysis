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

	
### Folder Structure
requirements.txt - packages required for running the code
main.py - the implementation code
data - folder with input data for main.py
figures - folder for .png line plots of individual participants and group
output - folder for .txt statistical report of individual and group
bcog200_paper.pdf - .pdf document of the final paper


## Testing Implementation detail
Install the package as noted in requirement.txt.
Directly Run the main.py (testing data is already in the data folder.)
The outcome plots will be in figures folder and statistical analysis in output folder.
The SVMs should originally be iterated for 1000 times, for the sake of testing, the current code
is written at 10 times iteration. 
note: iteration time can be modified in line 234 iteration_times = 10.



