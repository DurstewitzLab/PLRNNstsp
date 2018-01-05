function [Z,P,Zf]=PF_PLRNN(X,A,W,S,Inp,mu0,B,G,h,npart)
%
% please cite (& consult for further details): Durstewitz (2017) PLoS Comp Biol
% 
% implements state estimation by **particle filtering** for PLRNN system
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
% Z: (Mxnpart)xT matrix of npart state samples ('particles') at each time t
% P: npartxT matrix of normalized posterior densities ('weights')
%    p(X|Z)/sum_Z[p(X|Z)] for all samples 
% Zf: filtered states, i.e. sampled from Z according to the weights 


[~,T]=size(X);
M=length(h);
hh=repmat(h,1,npart);

%% initialize Z, Zf, P
Z=zeros(M*npart,T); P=zeros(npart,T); Zf=zeros(M*npart,T-1);
eps=randn(M*npart,T).*repmat(sqrt(diag(S)),npart,T);
Z(:,1)=repmat(mu0+Inp(:,1),npart,1)+eps(:,1);
zs=reshape(Z(:,1),M,npart);
mux_z=B*max(zs-hh,0);
P(:,1)=mvnpdf(repmat(X(:,1)',npart,1),mux_z',G);
Psum=sum(P(:,1));
if Psum<=0
    warning('t=1: all p=0; reset to p=1/ns');
    P(:,1)=ones(npart,1)./npart; Psum=1;
end;
P(:,1)=P(:,1)./Psum;

%% loop through all time points
for t=2:T
    % filter previous set of particles according to weights:
    pc=[0;cumsum(P(:,t-1))];
    y=histc(rand(npart,1),pc);
    k=find(y>0); zprev=[];
    for i=1:length(k), zprev=[zprev repmat(zs(:,k(i)),1,y(k(i)))]; end;
    Zf(:,t-1)=zprev(1:end)';
    % iterate state equation one step forward in t:
    zME=A*zprev+W*max(zprev-hh,0)+repmat(Inp(:,t),1,npart);
    zs=zME+reshape(eps(:,t),M,npart);
    Z(:,t)=zs(1:end)';
    % determine conditional mean of p(X|Z):
    mux_z=B*max(zs-hh,0);
    % compute new set of normalized weights at time step t:
    P(:,t)=mvnpdf(repmat(X(:,t)',npart,1),mux_z',G);
    Psum=sum(P(:,t));
    if Psum<=0
        warning(['t=' num2str(t) ': all p=0; reset to p=1/npart']);
        P(:,t)=ones(npart,1)./npart; Psum=1;
    end;
    P(:,t)=P(:,t)./Psum;
end;


%%
% (c) 2016 Durstewitz, Dept. Theoretical Neuroscience, Central Institute of
% Mental Health, Mannheim/ Heidelberg University
