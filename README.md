# AdaBoost.C2MIL

A new method combining adaboost and multiple instance learning. This method was designed for solving protein function prediction problems without residue labels. And it can be used for other similar purposes.

The file AdaBoost.C2MIL.py is the implementation of AdaBoost.C2MIL, and its input data file needs the instance labels, bag labels and bag identities as the first three columns with other features following.

Data format:
    Instance label | Bag label | Bag identity | Feature 1 | Feature 2 | ......
