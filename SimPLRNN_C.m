function [z,x]=SimPLRNN_C(A,W,S,Inp,mu0,B,G,h,C)
%
% *** Same as SimPLRNN, except that additional matrix C of weights for
% external regressors (i.e., inputs) is included
%
% simulates PLRNN as given in Durstewitz (2017) PLoS Comp Biol
% z_t = A z_t-1 + W max(z_t-1-h,0) + C Inp_t + e_t , e_t ~ N(0,S)
% x_t = B max(z_t-h,0) + nu_t , nu_t ~ N(0,G)
%
% INPUTS:
% A: MxM diagonal matrix 
% W: MxM off-diagonal matrix
% S: MxM diagonal covariance matrix (Gaussian process noise)
% Inp: KxT matrix of external inputs 
% mu0: Mx1 vector of initial values
% B: NxM matrix of regression weights
% G: NxN diagonal covariance matrix
% h: Mx1 vector of thresholds
% C: MxK matrix of regression weights multiplying with Inp
%
% OUTPUTS:
% Z: MxT matrix of (perturbed) latent states
% X: NxT matrix of observations generated from states

[~,T]=size(Inp);
[N,M]=size(B);
z=zeros(M,T);
z(:,1)=mvnrnd(mu0+C*Inp(:,1),S)';
eps=randn(M,T).*(sqrt(diag(S))*ones(1,T));
for t=2:T, z(:,t)=A*z(:,t-1)+W*max(z(:,t-1)-h,0)+C*Inp(:,t)+eps(:,t); end;
x=zeros(N,T);
nu=randn(N,T).*(sqrt(diag(G))*ones(1,T));
for t=1:T, x(:,t)=B*max(z(:,t)-h,0)+nu(:,t); end;


%%
% (c) 2016 Daniel Durstewitz, Dept. Theoretical Neuroscience, Central
% Institute of Mental Health, Heidelberg University
