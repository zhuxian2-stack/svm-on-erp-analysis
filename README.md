# BCOG200_Final_Project: Support Vector Machine on ERP data 


## Project Motivation:
The planned project is to create a pipeline for running SVM on ERP data; 
The project comes from a research question of 'can machine learning methods define a more precise time window for an Event-Related Potential component Nref, whose temporal boundaries are not clearly defined'. 
Current literature studied Nref within a very broad window of 400-1200ms.
The goal of this project is to replicate the N400 from Luck's paper on python, and the same model will be used with my first year project of NRef signal to cross-validate the Nref effect.


## Detail Information:
### Input Data Format:
This project accepts ERPlab EEG data format: xxx.set. 
The data format should be: for each participant folder, it has a cleaned .set file for each condition of the experiment (e.g. condition1.set, condition2.set). 


### Example Use Cases: 
As mentioned in the motivation, the prospect for this code is to navigate a better window for a component (NRef) with an ambiguous time window of interest, which gives a higher validity comparing to navigate the window by looking at the data pattern. 

	
### Folder Structure
* requirements.txt - packages required for running the code
* main.py - the implementation code
* data - folder with input data for main.py
* figures - folder for .png line plots of individual participants and group
* output - folder for .txt statistical report of individual and group
* Final paper - folder for bcog200_finalpaper.pdf



## Testing Implementation detail
* Create a virtual environment and install the package as noted in requirement.txt.
* Directly Run the main.py (testing data is already in the data folder.)
* The outcome plots will be in figures folder and statistical analysis in output folder.
* The SVMs should originally be iterated for 1000 times, for the sake of testing, the current code is written at 10 times iteration. 
  * note: iteration time can be modified in line 234 iteration_times = 10.



