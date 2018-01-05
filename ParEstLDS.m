function [mu0,B,G,W,A,S,ELL]=ParEstLDS(Ez,V,h,X_,Inp_,XZsplit,S0)
%
% please cite (& consult for further details): Durstewitz (2017) PLoS Comp Biol
% 
% implements parameter estimation for LDS
% z_t = A z_t-1 + W (z_t-1 - h) + Inp_t + e_t , e_t ~ N(0,S)
% x_t = B (z_t - h) + nu_t , nu_t ~ N(0,G)
%
% NOTE_1: X_, Inp_ can be matrices or cell arrays. If provided as cell
% arrays, each cell is interpreted as a single 'trial'; info is aggregated
% across trials, but temporal gaps are assumed between
% trials, i.e. temporal breaks are 'inserted' between trials
%
% NOTE_2: LDS estimation was set up such as to be maximally comparable to
%         PLRNN estimation process
%
%
% REQUIRED INPUTS:
% Ez: MxT matrix of state expectancies as returned by StateEstLDS
% V: state covariance matrix E[zz']-E[z]E[z]'
% h: Mx1 vector of constants ('thresholds')
% X_: NxT matrix of observations, or cell array of observation matrices
% Inp_: MxT matrix of external inputs, or cell array of input matrices 
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
% ELL: expected (complete data) log-likelihood


eps=1e-5;   % minimum variance allowed for in noise covariance


if iscell(X_), X=X_; Inp=Inp_; else X{1}=X_; Inp{1}=Inp_; end;
ntr=length(X);
m=size(Ez,1);
N=size(X{1},1);
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


%% compute all expectancy sums across trials & time points
EH=zeros(m); E2=EH; E3=EH; E3pkk=EH; E3_=EH; EH1=EH; EHT=EH;
F1=zeros(N,m); F2=zeros(N,N); F3=zeros(m,m); F4=F3; F5=F3; F6=F3;
for i=1:ntr
    mt=(Lsum(i)+1:Lsum(i+1))';
    Ez0=Ez(mt);
    Ezizi0=Ezizi(mt,mt);
    
    F1=F1+X{i}(:,1)*(Ez0(1:m)-h)';
    F2=F2+X{i}(:,1)*X{i}(:,1)';
    F5=F5+Ez0(1:m)*Inp{i}(:,1)';
    F6=F6+Inp{i}(:,1)*Inp{i}(:,1)';
    for t=2:T(i)
        k0=(t-1)*m+1:t*m;   % t
        k1=(t-2)*m+1:(t-1)*m;   % t-1
        EH=EH+Ez0(k0)*h';   % t=2:T
        E2=E2+Ezizi0(k0,k1);
        E3=E3+Ezizi0(k1,k1);
        E3_=E3_+Ezizi0(k0,k0);
        F1=F1+X{i}(:,t)*(Ez0(k0)-h)';
        F2=F2+X{i}(:,t)*X{i}(:,t)';
        F3=F3+Inp{i}(:,t)*Ez0(k1)';
        F4=F4+Inp{i}(:,t)*(Ez0(k1)-h)';
        F5=F5+Ez0(k0)*Inp{i}(:,t)';
        F6=F6+Inp{i}(:,t)*Inp{i}(:,t)';
    end;
    
    E3pkk=E3pkk+Ezizi0(k0,k0);
    
    EH1=EH1+Ez0(1:m)*h';    % t=1
    EHT=EHT+Ez0(k0)*h';     % t=T
    
    % solve for trial-specific parameters mu0
    mu0{i}=Ez0(1:m)-Inp{i}(:,1);
end;
E3p=E3+E3pkk;
H=h*h';
EH1_T=EH+EH1;
EH1_T1=EH1_T-EHT;
E1=E3-EH1_T1-EH1_T1'+(Tsum(end)-ntr).*H;
E1p=E3p-EH1_T-EH1_T'+Tsum(end).*H;
E4=E3-EH1_T1';
E5=E2-EH;



%% solve for parameters {B,G} of observation model
if nargin>5 && ~isempty(XZsplit)
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
O=ones(m)-eye(m);   % assumes W to have only OFF-diag elements
P0=(E5-(diag(diag(E2))-diag(diag(F3)))*diag(diag(E3))^-1*E4'-F4).*O;
P1=diag(diag(E3))^-1*E4';
W=zeros(m);
for i=1:length(W)
    K=E1-E4(:,i)*P1(i,:);
    j=setdiff(1:length(W),i);
    R=K(j,j);
    W(i,j)=P0(i,j)*R^-1;
end;


%% solve for auto-regressive weights A
A=diag(diag(E2-W*E4-F3))*diag(diag(E3))^-1;   % assumes A to be diag


%% solve for process noise covariance S, or use provided fixed S0
if nargin>6 && ~isempty(S0), S=S0;
else
    H=zeros(m);
    for i=1:ntr
        k=Lsum(i)+(1:m);
        H=H+V(k,k)+mu0{i}*Inp{i}(:,1)'+Inp{i}(:,1)*mu0{i}';
    end;
    S=diag(diag(H+E3_'-F5-F5'+F6-E2*A'-A*E2' ...
        +A*E3'*A'-E5*W'-W*E5'+W*E1'*W'+A*E4'*W'+W*E4*A'+A*F3'+F3*A'+W*F4'+F4*W'))./sum(T);   % assumes S to be diag
end;


%% compute expected log-likelihood (if desired)
if nargout>6
    LL0=0;
    for i=1:ntr
        LL0=mu0{i}'*S^-1*mu0{i}+mu0{i}'*S^-1*Inp{i}(:,1)+ ...
            Inp{i}(:,1)'*S^-1*mu0{i}-Ez(Lsum(i)+(1:m))'*S^-1*mu0{i}-mu0{i}'*S^-1*Ez(Lsum(i)+(1:m));
    end;
    LL1=trace(S^-1*E3p)-trace(S^-1*F5)-trace(S^-1*F5')+trace(S^-1*F6)-trace(S^-1*A*E2')-trace(A'*S^-1*E2) ...
        +trace(A'*S^-1*A*E3)-trace(S^-1*W*E5')-trace(W'*S^-1*E5)+trace(W'*S^-1*W*E1) ...
        +trace(A'*S^-1*W*E4)+trace(W'*S^-1*A*E4')+trace(A'*S^-1*F3)+trace(S^-1*A*F3') ...
        +trace(W'*S^-1*F4)+trace(S^-1*W*F4');
    LL2=trace(G^-1*F2)+trace(B'*G^-1*B*E1p)-trace(B'*G^-1*F1)-trace(G^-1*B*F1');
    ELL=-1/2*(LL0+LL1+LL2+sum(T)*log(det(G))+sum(T)*log(det(S)));
end;

if ~iscell(X_), mu0=cell2mat(mu0); end;



%%
% (c) 2017 Daniel Durstewitz, Dept. Theoretical Neuroscience, Central
% Institute of Mental Health, Heidelberg University
