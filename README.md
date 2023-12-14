# TCR-H
 TCR-H

TCR-H is a supervised classification ML model that can be used in four different settings of the training dataset
1) **Random Split**: Peptides and TCRs in test dataset that are already seen in training
2) **Epitope Hard Split**: Peptides in the test dataset that are not seen in training
3) **TCR Hard Split**: TCRs in the test dataset that are not seen in training
4) **Strict Split**: Peptides and TCRs in test dataset that are not seen in training

# Requirements
   python >= 3.0.0, numpy, pandas, joblib, matplotlib, scikit-learn

# Features Extraction
All the features of the TCR CDR3bs and the associated peptide sequences features to be calculated using peptides package in R (Osorio et al., 2015) or in python using [https://github.com/althonos/peptides.py]


# Train a new model and make prediction
   # Usage

For Random Split:

python TCR-H-randomsplit.py 

Given:

Inputfile should be a comma-separated file (CSV) containing columns for TCR CDR3b, epitope, label (0 for non-binding and 1 for binding) and all the features of CDR3b and peptide sequences calculated of both training and test data (see example training/test data).

For Hard/Strict Splits:

python TCR-H-hard.py 

Given:

training datafile should be a comma-separated file (CSV) containing columns for TCR CDR3b, epitope, label (0 for non-binding and 1 for binding) and all the features of CDR3b and peptide sequences calculated. See provided example training data 

test datafile should be a CSV file containing columns for TCR CDR3b, epitope, label (0 for non-binding and 1 for binding) and all the features of CDR3b and peptide sequences calculated. See example test file.

The prediction output is printed in the  output file provided.

  # Contact
  
For more questions/feedback, Please post an Issue.

# References:
Osorio, D., Rondón-Villarreal, P., & Torres, R. (2015). Peptides: A package for data mining of antimicrobial peptides. R Journal, 7(1), 4–14. https://doi.org/10.32614/rj-2015-001

# Citation:
 Rajitha Rajeshwar T., Omar Demerdash, Jeremy C. Smith TCR-H: Machine Learning Prediction of T-cell Receptor Epitope Binding on Unseen Datasets _bioRxiv_ 2023.11.28.569077
   
