function [EzSim,EphiziSim,EzizjSim,EphizijSim,EziphizjSim]=PF_ExpVal(X,A,W,S,Inp,mu0,B,G,h,npart)
%
% please cite (& consult for further details): Durstewitz (2017) PLoS Comp Biol
% 
% returns all state expectancies based on PF for PLRNN system
% z_t = A z_t-1 + W max(z_t-1-h,0) + Inp_t + e_t , e_t ~ N(0,S)
% x_t = B max(z_t-h,0) + nu_t , nu_t ~ N(0,G)
%
%
% REQUIRED INPUTS:
% X: NxT matrix of observations
% A: MxM diagonal matrix 
% W: MxM off-diagonal matrix
% S: MxM diagonal covariance matrix (Gaussian process noise)
% Inp: MxT matrix of external inputs
% mu0: Mx1 vector of initial values
% B: NxM matrix of regression weights
% G: NxN diagonal covariance matrix
% h: Mx1 vector of thresholds
% npart: number of particles (samples)
%
% OUTPUTS:
% EzSim: numerically estimated ('simulated') state expectations
% EphiziSim: E[phi(z)]
% EzizjSim: E[zz']
% EphizijSim: E[phi(z) phi(z)']
% EziphizjSim: E[z phi(z)']


%% run particle filter
[Z,P,Zf]=PF_PLRNN(X,A,W,S,Inp,mu0,B,G,h,npart);


%% from PF output, determine actual expectancy values of states 
% & functions phi(Z) of states
T=size(X,2);
M=length(mu0);
hh=repmat(h,npart,T);
phiZ=max(0,Z-hh); phiZf=max(0,Zf-hh(:,1:T-1));
EzSim=zeros(M,T); EphiziSim=zeros(M,T);
for k=1:M
    EzSim(k,:)=sum(Z(k:M:end,:).*P);
    EphiziSim(k,:)=sum(phiZ(k:M:end,:).*P);
end;
EzizjSim=zeros(M*T); EphizijSim=zeros(M*T); EziphizjSim=zeros(M*T);
% compute all covariance matrices:
k0=1:M;
zt=reshape(Z(:,1),M,npart); phizt=reshape(phiZ(:,1),M,npart);
pt=repmat(P(:,1)',M,1);
EzizjSim(k0,k0)=(pt.*zt)*zt';
EphizijSim(k0,k0)=(pt.*phizt)*phizt';
EziphizjSim(k0,k0)=(pt.*zt)*phizt';
for t=2:T
    k0=(t-1)*M+1:t*M;   % t
    k1=(t-2)*M+1:(t-1)*M;   % t-1
    zt0=reshape(Z(:,t),M,npart); phizt0=reshape(phiZ(:,t),M,npart);
    zt1=reshape(Zf(:,t-1),M,npart); phizt1=reshape(phiZf(:,t-1),M,npart);
    pt=repmat(P(:,t)',M,1);
    % same t
    EzizjSim(k0,k0)=(pt.*zt0)*zt0';
    EphizijSim(k0,k0)=(pt.*phizt0)*phizt0';
    EziphizjSim(k0,k0)=(pt.*zt0)*phizt0';
    % t, t-1
    EzizjSim(k0,k1)=(pt.*zt0)*zt1'; EzizjSim(k1,k0)=EzizjSim(k0,k1)';
    EphizijSim(k0,k1)=(pt.*phizt0)*phizt1'; EphizijSim(k1,k0)=EphizijSim(k0,k1)';
    EziphizjSim(k0,k1)=(pt.*zt0)*phizt1'; EziphizjSim(k1,k0)=EziphizjSim(k0,k1)';
end;


%%
% (c) 2016 Durstewitz, Dept. Theoretical Neuroscience, Central Institute of
% Mental Health, Mannheim/ Heidelberg University
