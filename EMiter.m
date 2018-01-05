function [mu0_est,B_est,G_est,W_est,A_est,S_est,Ezi,Vest,Ephizi,Ephizij,Eziphizj,LL,Err,NumIt]= ...
    EMiter(CtrPar,A_est,W_est,S_est,Inp,mu0_est,B_est,G_est,h,X,XZspl)
%
% please cite (& consult for further details): Durstewitz (2017) PLoS Comp Biol
% 
% implements EM iterations for PLRNN system
% z_t = A z_t-1 + W max(z_t-1-h,0) + Inp_t + e_t , e_t ~ N(0,S)
% x_t = B max(z_t-h,0) + nu_t , nu_t ~ N(0,G)
%
% NOTE: Inp, mu0_est, X can be matrices or cell arrays. If provided as cell
% arrays, each cell is interpreted as a single 'trial'; trials are
% concatenated for estimation, but temporal gaps are assumed between
% trials, i.e. temporal breaks are 'inserted' between trials
%
%
% REQUIRED INPUTS:
% CtrPar=[tol MaxIter tol2 eps flipOnIt]: vector of control parameters:
% -- tol: relative (to first LL value) tolerated increase in log-likelihood
% -- MaxIter: maximum number of EM iterations allowed
% -- tol2: relative error tolerance in state estimation (see StateEstPLRNN) 
% -- eps: singularity parameter in StateEstPLRNN
% -- flipOnIt: parameter that controls switch from single (i<=flipOnIt) to 
%              full (i>flipOnIt) constraint flipping in StateEstPLRNN
% A_est: initial estimate of MxM diagonal matrix of auto-regressive weights
% W_est: initial estimate of MxM off-diagonal matrix of interaction weights
% S_est: initial estimate of MxM diagonal process covariance matrix
%        (assumed to be constant here)
% Inp: MxT matrix of external inputs, or cell array of input matrices 
% mu0_est: initial estimate of Mx1 vector of initial values, or cell array of Mx1 vectors
% B_est: initial estimate of NxM matrix of regression weights
% G_est: initial estimate of NxN diagonal observation covariance matrix
% h: Mx1 vector of (fixed) thresholds
% X: NxT matrix of observations, or cell array of observation matrices
%
% OPTIONAL INPUTS:
% XZspl: vector [Nx Mz] which allows to assign certain states only to 
%        certain observations; specifically, the first Mz state var are
%        assigned to the first Nx obs, and the remaining state var to the
%        remaining obs; (1:Mz)-->(1:Nx), (Mz+1:end)-->(Nx+1:end)
%
%
% OUTPUTS:
% final estimates of network parameters {mu0_est,B_est,G_est,W_est,A_est,S_est}
% Ezi: MxT matrix of state expectancies as returned by StateEstPLRNN
% Vest: estimated state covariance matrix E[zz']-E[z]E[z]'
% Ephizi: E[phi(z)]
% Ephizij: E[phi(z) phi(z)']
% Eziphizj: E[z phi(z)']
% LL: log-likelihood (vector) as function of EM iteration 
% Err: final error returned by StateEstPLRNN
% NumIt: total number of EM + mode-search iterations


if nargin<11, XZspl=[]; end;

tol=CtrPar(1);
MaxIter=CtrPar(2);
tol2=CtrPar(3);
eps=CtrPar(4);
flipOnIt=CtrPar(5);

%% EM loop
i=1; LLR=1e8; LL=[]; Ezi=[]; NumIt=0;
while (i==1 || LLR>tol*abs(LL(1)) || LLR<0) && i<MaxIter

    % E-step
    if i>flipOnIt, flipAll=true; else flipAll=false; end;
    [Ezi,U,~,Err]=StateEstPLRNN(A_est,W_est,S_est,Inp,mu0_est,B_est,G_est,h,X,Ezi,[],tol2,eps,flipAll);
    [Ephizi,Ephizij,Eziphizj,Vest]=ExpValPLRNN(Ezi,U,h);
    
    NumIt=NumIt+length(Err);    % total # of iterations (EM x mode search)

    % M-step
    [mu0_est,B_est,G_est,W_est,A_est,S_est]=ParEstPLRNN(Ezi,Vest,Ephizi,Ephizij,Eziphizj,X,Inp,XZspl,S_est);

    % compute log-likelihood (alternatively, use ELL output from ParEstPLRNN)
    LL(i)=LogLikePLRNN(A_est,W_est,S_est,Inp,mu0_est,B_est,G_est,h,X,Ezi);
    
    if i>1, LLR=LL(i)-LL(i-1); else LLR=1e8; end;   % LL ratio 
    i=i+1;

end;
disp(['LL= ' num2str(LL(end)) ', # iterations= ' num2str(i-1)]);


%%
% (c) 2016 Daniel Durstewitz, Dept. Theoretical Neuroscience, Central
% Institute of Mental Health, Heidelberg University
