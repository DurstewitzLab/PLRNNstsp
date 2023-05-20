#  State inference and parameter estimation in piecewise-linear recurrent neural network (PLRNN) models 

Matlab code for state inference and parameter estimation in piecewise-linear recurrent neural network (PLRNN) models

This folder contains the MatLab code and data files from

[Durstewitz, D (2017) *A State Space Approach for Piecewise-Linear Recurrent Neural Networks
   for Identifying Computational Dynamics from Neural Measurements.* PLoS Computational Biology.](http://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005542)

Copyright: © 2017 Daniel Durstewitz.

This package is distributed under the terms of the GNU GPLv3 & Creative Commons Attribution License.
Please credit the source and **cite the reference above** when using the code in any from of publication.


--- Main code for PLRNN estimation:  
runPLRNN_WMexample.m : illustrates how to use the code  
EMiter.m : EM iterations for PLRNN  
StateEstPLRNN.m : state inference for PLRNN  
ExpValPLRNN.m : compute all other PLRNN expectancies  
ParEstPLRNN.m : parameter estimation for PLRNN  
LogLikePLRNN.m: log-likelihood  
SimPLRNN.m : simulate PLRNN  
runPLRNN_DataExample.m : run PLRNN estimation on ACC MSU recording data  

--- same code but including weight matrix C for external regressors:  
runPLRNN_C_example.m  
StateEstPLRNN_C.m  
ParEstPLRNN_C.m  
LogLikePLRNN_C.m  
SimPLRNN_C.m  

--- code for PLRNN state estimation through particle filtering:  
PF_ExpVal.m  
PF_PLRNN.m  
runPF_example.m  

--- code for linear dynamical system (LDS) derived from PLRNN:  
EMiter_LDS.m  
StateEstLDS.m  
ParEstLDS.m  
LogLikeLDS.m  

--- helper functions:  
invblocktridiag.m (created by Hazem Toutounji)  
ExtractBlockDiag.m  
GraphExpec.m  

--- parameter files for illustrative examples:  
PLRNNwmParam.mat  
PLRNNoscParam.mat  

--- MSU recording data:  
del_alt_11_26_KDE05.mat  
- data are from Hyman JM et al., 2013, Cerebral Cortex 23: 1257-1268.

del_alt_11_26_KDE01.mat = same data at higher temporal resolution (100 ms binwidth)
