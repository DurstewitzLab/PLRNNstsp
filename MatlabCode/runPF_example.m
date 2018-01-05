%% illustrates state estimation for PLRNN by particle filter (PF) 
% reproduces Fig. 2 from Durstewitz (2017), PLoS Comp Biol (please see this
% ref. for further details)
%
%
clear all
close all


%% set parameters of PLRNN (limit cycle example)
load PLRNNoscParam
M=length(h);
ntr=50;
Inp=repmat(InpVal,1,ntr);


%% simulate perturbed system
randn('state',0);
[Z,X]=SimPLRNN(A,W,S,Inp,mu0,B,G,h);
 

%% compute state expectancies through paticle filtering
n_particles=10^5;   % # of particles to use
[EzSim,EphiziSim,EzizjSim,EphizijSim,EziphizjSim]=PF_ExpVal(X,A,W,S,Inp,mu0,B,G,h,n_particles);


%% compute state expectancies by mode-search algo
flipAll=true;   % flip all violated constraints on each iteration 
[EzEst,U]=StateEstPLRNN(A,W,S,Inp,mu0,B,G,h,X,[],[],[],[],flipAll);  % estimate Hessian U
[EphiziEst,EphizijEst,EziphizjEst,Vest]=ExpValPLRNN(EzEst,U,h);  % estimate all other state expectancies 
EzizjEst=Vest+EzEst(1:end)'*EzEst(1:end);   % compute E[zz'] from state cov-matrix



%% graph overall correlations between simulated and estimated state/cov expectations
figure(2), hold off cla
GraphExpec(EzSim(1:end)',EzEst(1:end)',1,'z',[1 -1]);
y=diag(EzizjSim); yEst=diag(EzizjEst);
GraphExpec(y,yEst,2,'z^2',[5 -5]);
y=ExtractBlockDiag(EzizjSim,2*M); yEst=ExtractBlockDiag(EzizjEst,2*M);
GraphExpec(y',yEst',3,'z_iz_j',[5 -5]);
GraphExpec(EphiziSim(1:end)',EphiziEst,4,'\Phi(z)',[1 -1]);
y=diag(EphizijSim); yEst=diag(EphizijEst);
GraphExpec(y,yEst,5,'\Phi(z)^2',[5 -5]);
y=ExtractBlockDiag(EphizijSim,2*M); yEst=ExtractBlockDiag(EphizijEst,2*M);
GraphExpec(y',yEst',6,'\Phi(z_i)\Phi(z_j)',[5 -5]);
y=diag(EziphizjSim); yEst=diag(EziphizjEst);
GraphExpec(y,yEst,7,'z\Phi(z)',[5 -5]);
y=ExtractBlockDiag(EziphizjSim,2*M); yEst=ExtractBlockDiag(EziphizjEst,2*M);
GraphExpec(y',yEst',8,'z_i\Phi(z_j)',[5 -5]);


%%
% (c) 2016 Daniel Durstewitz, Dept. Theoretical Neuroscience, Central
% Institute of Mental Health, Heidelberg University
