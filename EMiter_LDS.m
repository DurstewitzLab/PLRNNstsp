function [mu0_est,B_est,G_est,W_est,A_est,S_est,Ezi,Vest,LL]= ...
    EMiter_LDS(CtrPar,A_est,W_est,S_est,Inp,mu0_est,B_est,G_est,h,X,XZspl)
%
% please cite (& consult for further details): Durstewitz (2017) PLoS Comp Biol
% 
% implements EM iterations for LDS
% z_t = A z_t-1 + W (z_t-1 - h) + Inp_t + e_t , e_t ~ N(0,S)
% x_t = B (z_t - h) + nu_t , nu_t ~ N(0,G)
%
% NOTE: Inp, mu0_est, X can be matrices or cell arrays. If provided as cell
% arrays, each cell is interpreted as a single 'trial'; trials are
% concatenated for estimation, but temporal gaps are assumed between
% trials, i.e. temporal breaks are 'inserted' between trials
%
%
% REQUIRED INPUTS:
% CtrPar=[tol MaxIter __ eps]: vector of control parameters:
% -- tol: relative (to first LL value) tolerated increase in log-likelihood
% -- MaxIter: maximum number of EM iterations allowed
% -- __: 3rd param. is irrelevant for LDS (just included to parallel PLRNN files) 
% -- eps: singularity parameter in StateEstLDS
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
% Ezi: MxT matrix of state expectancies as returned by StateEstLDS
% Vest: estimated state covariance matrix E[zz']-E[z]E[z]'
% LL: log-likelihood (vector) as function of EM iteration 


if nargin<11, XZspl=[]; end;

tol=CtrPar(1);
MaxIter=CtrPar(2);
eps=CtrPar(4);


%% EM loop
i=1; LLR=1e8; LL=[]; Ezi=[];
while (i==1 || LLR>tol*abs(LL(1)) || LLR<0) && i<MaxIter

    % E-step
    [Ezi,Vest]=StateEstLDS(A_est,W_est,S_est,Inp,mu0_est,B_est,G_est,h,X,eps);

    % M-step
    [mu0_est,B_est,G_est,W_est,A_est,S_est]=ParEstLDS(Ezi,Vest,h,X,Inp,XZspl,S_est);

    % compute log-likelihood (alternatively, use ELL output from ParEstLDS)
    LL(i)=LogLikeLDS(A_est,W_est,S_est,Inp,mu0_est,B_est,G_est,h,X,Ezi);
    
    if i>1, LLR=LL(i)-LL(i-1); else LLR=1e8; end;    % LL ratio 
    i=i+1;

end;
disp(['fin LL = ' num2str(LL(end)) ' , #iterations = ' num2str(i-1)]);


%%
% (c) 2017 Daniel Durstewitz, Dept. Theoretical Neuroscience,
% Central Institute of Mental Health Mannheim, Heidelberg University
