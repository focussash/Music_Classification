To run each of the classifiers, make sure the python codes are placed in a folder (any folder) in which the folders "training" and "test" are stored.
Further, the labels.csv file which contain training data labels and Output.csv file which the codes are set to output to must also be in the same folder as the codes
If you want to train from a different set of files, the relevant file names need to be changed accordingly
Invoking the classifiers will by default learn from the files in training folder and try to classify files in test folder

Note that the paths notation is different across operating systems, specifically whether it uses "/" or "\"
I included both versions of the codes, you can also switch by uncommenting the (commented) other path line and comment out the current one in use