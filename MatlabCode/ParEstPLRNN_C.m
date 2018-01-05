function [mu0,B,G,W,A,S,C,ELL]=ParEstPLRNN_C(Ez,V,Ephizi,Ephizij,Eziphizj,X_,Inp_,XZsplit,S0)
%
% *** Same as ParEstPLRNN, except that additional matrix C of weights for
% external regressors (i.e., inputs) is estimated
%
% please cite (& consult for further details): Durstewitz (2017) PLoS Comp Biol
% 
% implements parameter estimation for PLRNN system
% z_t = A z_t-1 + W max(z_t-1-h,0) + C Inp_t + e_t , e_t ~ N(0,S)
% x_t = B max(z_t-h,0) + nu_t , nu_t ~ N(0,G)
%
% NOTE: X_, Inp_ can be matrices or cell arrays. If provided as cell
% arrays, each cell is interpreted as a single 'trial'; info is aggregated
% across trials, but temporal gaps are assumed between
% trials, i.e. temporal breaks are 'inserted' between trials
%
%
% REQUIRED INPUTS:
% Ez: MxT matrix of state expectancies as returned by StateEstPLRNN
% V: state covariance matrix E[zz']-E[z]E[z]'
% Ephizi: E[phi(z)]
% Ephizij: E[phi(z) phi(z)']
% Eziphizj: E[z phi(z)']
% X_: NxT matrix of observations, or cell array of observation matrices
% Inp_: KxT matrix of external inputs, or cell array of input matrices 
%
% OPTIONAL INPUTS:
% XZsplit: vector [Nx Mz] which allows to assign certain states only to 
%          certain observations; specifically, the first Mz state var are
%          assigned to the first Nx obs, and the remaining state var to the
%          remaining obs; (1:Mz)-->(1:Nx), (Mz+1:end)-->(Nx+1:end)
% S0: fix process noise-cov matrix to S0
%
% OUTPUTS:
% mu0: Mx1 vector of initial values, or cell array of Mx1 vectors
% B: NxM matrix of regression weights
% G: NxN diagonal covariance matrix
% W: MxM off-diagonal matrix of interaction weights
% A: MxM diagonal matrix of auto-regressive weights
% S: MxM diagonal covariance matrix (Gaussian process noise)
% C: MxK matrix of regression weights multiplying with Kx1 Inp
% ELL: expected (complete data) log-likelihood


eps=1e-5;   % minimum variance allowed for in S and G

if iscell(X_), X=X_; Inp=Inp_; else X{1}=X_; Inp{1}=Inp_; end;
ntr=length(X);
m=size(Ez,1);
N=size(X{1},1);
Minp=size(Inp{1},1);
T=cell2mat(cellfun(@size,X,'UniformOutput',false)'); T=T(:,2);
Tsum=cumsum([0 T']);
Lsum=Tsum.*m;


%% compute E[zz'] from state cov matrix V
Ez=Ez(1:end)';
Ezizi=sparse(m*sum(T),m*sum(T));
for i=1:ntr
    for t=Tsum(i)+1:(Tsum(i+1)-1)
        k0=(t-1)*m+1:t*m;
        k1=t*m+1:(t+1)*m;
        Ezizi(k0,[k0 k1])=V(k0,[k0 k1])+Ez(k0)*Ez([k0 k1])';
        Ezizi(k1,k0)=Ezizi(k0,k1)';
    end;
    Ezizi(k1,k1)=V(k1,k1)+Ez(k1)*Ez(k1)';
end;


%% compute all expectancy sums across trials & time points (eq. 16)
E1=zeros(m); E2=E1; E3=E1; E4=E1; E5=E1; E1pkk=E1; E3pkk=E1; E3_=E1;
F1=zeros(N,m); F2=zeros(N,N); F3=zeros(Minp,m); F4=F3; F5_=zeros(m,Minp); F6_=zeros(Minp,Minp);
f5_1=F5_; f6_1=F6_;
for i=1:ntr
    mt=(Lsum(i)+1:Lsum(i+1))';
    Ez0=Ez(mt);
    Ephizi0=Ephizi(mt);
    Ezizi0=Ezizi(mt,mt);
    Ephizij0=Ephizij(mt,mt);
    Eziphizj0=Eziphizj(mt,mt);
    
    F1=F1+X{i}(:,1)*Ephizi0(1:m)';
    F2=F2+X{i}(:,1)*X{i}(:,1)';
    f5_1=f5_1+Ez0(1:m)*Inp{i}(:,1)';
    f6_1=f6_1+Inp{i}(:,1)*Inp{i}(:,1)';
    for t=2:T(i)
        k0=(t-1)*m+1:t*m;   % t
        k1=(t-2)*m+1:(t-1)*m;   % t-1
        E1=E1+Ephizij0(k1,k1);
        E2=E2+Ezizi0(k0,k1);
        E3=E3+Ezizi0(k1,k1);
        E3_=E3_+Ezizi0(k0,k0);
        E4=E4+Eziphizj0(k1,k1)';
        E5=E5+Eziphizj0(k0,k1);
        F1=F1+X{i}(:,t)*Ephizi0(k0)';
        F2=F2+X{i}(:,t)*X{i}(:,t)';
        F3=F3+Inp{i}(:,t)*Ez0(k1)';
        F4=F4+Inp{i}(:,t)*Ephizi0(k1)';
        F5_=F5_+Ez0(k0)*Inp{i}(:,t)';
        F6_=F6_+Inp{i}(:,t)*Inp{i}(:,t)';
    end;
    E1pkk=E1pkk+Ephizij0(k0,k0);
    if nargout>6, E3pkk=E3pkk+Ezizi0(k0,k0); end;
end;
E1p=E1+E1pkk;
F5=F5_+f5_1;
F6=F6_+f6_1;


%% solve for parameters {B,G} of observation model
if nargin>7 && ~isempty(XZsplit)
    Nx=XZsplit(1); Mz=XZsplit(2);
    F1_=F1; F1_(1:Nx,Mz+1:end)=0; F1_(Nx+1:end,1:Mz)=0;
    E1p_=E1p; E1p_(1:Mz,Mz+1:end)=0; E1p_(Mz+1:end,1:Mz)=0;
    B=F1_*E1p_^-1;
    G=diag(max(diag(F2-F1_*B'-B*F1_'+B*E1p_'*B')./sum(T),eps));   % assumes G to be diag
else
    B=F1*E1p^-1;
    G=diag(max(diag(F2-F1*B'-B*F1'+B*E1p'*B')./sum(T),eps));   % assumes G to be diag
end;


%% solve for interaction weight matrix W
P0=F3'*F6_^-1*F4;
L0=diag(diag(E3-F3'*F6_^-1*F3));
L1=diag(diag(E2-F5_*F6_^-1*F3));
L2=P0'-E4;
K=L0^-1*(E4'-P0);
Y=E5-F5_*F6_^-1*F4+L1*L0^-1*(P0-E4');
P1=E1-F4'*F6_^-1*F4;
W=zeros(m);
for i=1:length(W)
    j=setdiff(1:length(W),i);
    R=P1+L2(:,i)*K(i,:);
    W(i,j)=Y(i,j)*R(j,j)^-1;
end;

%% solve for auto-regressive weights A
A=(L1+diag(diag(W*L2)))*L0^-1;   % assumes A to be diag

%% solve for external regressor weights C
C=(F5_-W*F4'-A*F3')*F6_^-1;

%% solve for trial-specific parameters mu0
for i=1:ntr
    mt=(Lsum(i)+1:Lsum(i+1))';
    Ez0=Ez(mt);
    mu0{i}=Ez0(1:m)-C*Inp{i}(:,1);
end;

%% solve for process noise covariance S, or use provided fixed S0
if nargin>8 && ~isempty(S0), S=S0;
else
    H=zeros(m);
    for i=1:ntr
        k=Lsum(i)+(1:m);
        H=H+Ezizi(k,k)-Ez(k)*mu0{i}'-mu0{i}*Ez(k)'+mu0{i}*mu0{i}'+mu0{i}*Inp{i}(:,1)'*C'+C*Inp{i}(:,1)*mu0{i}';
    end;
    S=diag(diag(H+E3_'-F5*C'-C*F5'+C*F6*C'-E2*A'-A*E2' ...
        +A*E3'*A'-E5*W'-W*E5'+W*E1'*W'+A*E4'*W'+W*E4*A'+A*F3'*C'+C*F3*A'+W*F4'*C'+C*F4*W'))./sum(T);   % assumes S to be diag
end;


%% compute expected log-likelihood (if desired)
if nargout>6
    E3p=E3+E3pkk;
    LL0=0;
    for i=1:ntr
        LL0=mu0{i}'*S^-1*mu0{i}+mu0{i}'*S^-1*C*Inp{i}(:,1)+ ...
            Inp{i}(:,1)'*C'*S^-1*mu0{i}-Ez(Lsum(i)+(1:m))'*S^-1*mu0{i}-mu0{i}'*S^-1*Ez(Lsum(i)+(1:m));
    end;
    LL1=trace(S^-1*E3p)-trace(S^-1*F5*C')-trace(S^-1*C*F5')+trace(S^-1*C*F6*C')-trace(S^-1*A*E2')-trace(A'*S^-1*E2) ...
        +trace(A'*S^-1*A*E3)-trace(S^-1*W*E5')-trace(W'*S^-1*E5)+trace(W'*S^-1*W*E1) ...
        +trace(A'*S^-1*W*E4)+trace(W'*S^-1*A*E4')+trace(A'*S^-1*C*F3)+trace(S^-1*A*F3'*C') ...
        +trace(W'*S^-1*C*F4)+trace(S^-1*W*F4'*C');
    LL2=trace(G^-1*F2)+trace(B'*G^-1*B*E1p)-trace(B'*G^-1*F1)-trace(G^-1*B*F1');
    ELL=-1/2*(LL0+LL1+LL2+sum(T)*log(det(G))+sum(T)*log(det(S)));
end;

if ~iscell(X_), mu0=cell2mat(mu0); end;


%%
% (c) 2016 Daniel Durstewitz, Dept. Theoretical Neuroscience, Central
% Institute of Mental Health, Heidelberg University
