# Information-from-dynamical-landscape
Basic code necessary for the reproduction of the results in "Measuring developmental information encoded by a dynamical landscape"

The two folders correspond to the two exemplar systems presented.

The folder **NeurMesoDiff** correspondonds to the Neural Mesodermal differentiation system. 
The file _ProportionsAll.mat_ contains the simulated proportions for the 10000 paramater values considered in the matrix _Props_. The first dimension corresponds to experimental condition. The second to parameter vector and the third contains the 6 fates for the 6 measurement time-points. They are ordered as 'EPI','AN', 'CE', 'PN', 'M', 'UT' for day 2.5, then the same fates for day 3, the same for day 3.5 and so on.

The script _script_ComputeTest_1000_AnalyseSeparating11Cond_Prop_AllTime_M.m_ contains the necessary code to compute potency for this system. One can change de number of proportion vectors used for training with the variable _numparamTrain_, the number of repetitions of the process with _numRepetitions_ the conditions considered with the variable _Cases_ and the classifier by changing the function _fitcecoc_ for another one.


The folder **WormVulvaPatt** correspondonds to the Worm Vulva Patterning system. 
The file _VulvalDevelopmentPropsClass.mat_ contains the simulated proportions for the 20000 paramater values considered in the matrix _MatrixProps_. The first 12 columns correspond to the different fates in the different cells and the 13th column specifies the mutant being considered.

The script _script_ComputeTest_9cond.m_ contains the necessary code to compute potency for this system. One can change de number of proportion vectors used for training with the variable _numparamTrain_, the number of repetitions of the process with _numRepetitions_ the conditions considered with the variable _conditions_ and the classifier by changing the function _fitcecoc_ for another one.

The script _script_ComputeTest_9cond_Capacity.m_ contains the necessary code to compute capacity for this system. One can change de number of proportion vectors used for training with the variable _numparamTrain_, the number of repetitions of the process with _numRepetitions_ the conditions considered with the variable _conditions_ and the classifier by changing the function _fitcecoc_ for another one.

