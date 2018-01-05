function [z,U,d,Err]=StateEstPLRNN(A,W,S,Inp_,mu0_,B,G,h,X_,z0,d0,tol,eps,flipAll)
%
% please cite (& consult for further details): Durstewitz (2017) PLoS Comp Biol
% 
% implements state estimation for PLRNN system
% z_t = A z_t-1 + W max(z_t-1-h,0) + Inp_t + e_t , e_t ~ N(0,S)
% x_t = B max(z_t-h,0) + nu_t , nu_t ~ N(0,G)
%
% NOTE: Inp_, mu0_, X_ can be matrices or cell arrays. If provided as cell
% arrays, each cell is interpreted as a single 'trial'; trials are
% concatenated for estimation, but temporal gaps are assumed between
% trials, i.e. temporal breaks are 'inserted' between trials
%
%
% REQUIRED INPUTS:
% A: MxM diagonal matrix 
% W: MxM off-diagonal matrix
% S: MxM diagonal covariance matrix (Gaussian process noise)
% Inp_: MxT matrix of external inputs, or cell array of input matrices 
% mu0_: Mx1 vector of initial values, or cell array of Mx1 vectors
% B: NxM matrix of regression weights
% G: NxN diagonal covariance matrix
% h: Mx1 vector of thresholds
% X_: NxT matrix of observations, or cell array of observation matrices
%
% OPTIONAL INPUTS:
% z0: initial guess of state estimates provided as (MxT)x1 vector
% d0: initial guess of constraint settings provided as 1x(MxT) vector 
% tol: acceptable relative tolerance for error increases (default: 1e-2)
% eps: small number added to state covariance matrix for avoiding
%      singularities (default: 0)
% flipAll: flag which determines whether all constraints are flipped at
%          once on each iteration (=true) or whether only the most violating
%          constraint is flipped on each iteration (=false)
%
% OUTPUTS:
% z: estimated state expectations
% U: Hessian
% Err: final total threshold violation error
% d: final set of constraints (ie, for which z>h) 

if nargin<12 || isempty(tol), tol=1e-2; end;
if nargin<13, eps=[]; end;
if nargin<14, flipAll=false; end;

m=length(A);    % # of latent states

if iscell(X_), X=X_; Inp=Inp_; mu0=mu0_;
else X{1}=X_; Inp{1}=Inp_; mu0{1}=mu0_; end;
ntr=length(X);  % # of distinct trials


%% construct block-banded components of Hessian U0, U1, U2, and 
% vectors/ matrices v0, v1, V2, V3, V4, as specified in the objective 
% function Q(Z), eq. 7, in Durstewitz (2017)
u0=S^-1+A'*S^-1*A; K0=-A'*S^-1;
u2=W'*S^-1*W+B'*G^-1*B;
u1=W'*S^-1*A; K2=-W'*S^-1;
v2=-S^-1*W; v3=A'*S^-1*W; v4=B'*G^-1*B+W'*S^-1*W;
U0=[]; U2=[]; U1=[];
v0=[]; v1=[]; V2=[]; V3=[]; V4=[];
Tsum=0;
for i=1:ntr   % acknowledge temporal breaks between trials
    T=size(X{i},2); Tsum=Tsum+T;
    
    U0_=u0; KK0=K0; U2_=u2; U1_=u1; KK2=K2;
    for t=1:T-1
        U0_=blkdiag(U0_,u0); KK0=blkdiag(KK0,K0);
        U2_=blkdiag(U2_,u2);
        U1_=blkdiag(U1_,u1); KK2=blkdiag(KK2,K2);
    end;
    KK0=blkdiag(KK0,K0);
    kk=(T-1)*m+1:T*m; U0_(kk,kk)=S^-1;
    U0_=U0_+KK0(m+1:end,1:T*m);
    KK0=KK0'; U0_=U0_+KK0(1:T*m,m+1:end);
    U2_(kk,kk)=B'*G^-1*B;
    U1_(kk,kk)=0; KK2=blkdiag(KK2,K2);
    U1_=U1_+KK2(m+1:end,1:T*m);
    U0=sparse(blkdiag(U0,U0_)); U2=sparse(blkdiag(U2,U2_)); U1=sparse(blkdiag(U1,U1_));
    
    vka=S^-1*Inp{i}; vka(:,1)=vka(:,1)+S^-1*mu0{i};
    vkb=A'*S^-1*Inp{i}(:,2:T);
    v0_=(vka(1:end)-[vkb(1:end) zeros(1,m)])';
    vka=B'*G^-1*X{i};
    vkb=-W'*S^-1*Inp{i}(:,2:T);
    v1_=(vka(1:end)+[vkb(1:end) zeros(1,m)])';
    V2_=v2; V3_=v3; V4_=v4;
    for t=1:T-1, V2_=blkdiag(V2_,v2); V3_=blkdiag(V3_,v3); V4_=blkdiag(V4_,v4); end;
    V2_(1:m,1:m)=0; V3_(kk,kk)=0; V4_(kk,kk)=B'*G^-1*B;
    v0=[v0;v0_]; v1=[v1;v1_];
    V2=sparse(blkdiag(V2,V2_)); V3=sparse(blkdiag(V3,V3_)); V4=sparse(blkdiag(V4,V4_));
end;


%% initialize states and constraint vector
hh=repmat(h,Tsum,1);
n=1; idx=0; k=[];
if nargin>9 && ~isempty(z0), z=z0(1:end)'; else z=hh+randn(m*Tsum,1); end;
if nargin>10 && ~isempty(d0), d=d0; else d=zeros(1,m*Tsum); d(z>hh)=1; end;
Err=1e16;
y=rand(m*Tsum,1); LL=d*y;  % define arbitrary projection vector for detecting already visited solutions 
% alternative: LL=bin2dec(num2str(d)), but doesn't work for large numbers (>63bit)
U=[]; dErr=-1e8;

% % compute initial log-likelihood (if desired)
% LogLike=[];
% d0=zeros(1,m*Tsum); d0(z>hh)=1; D0=spdiags(d0',0,m*Tsum,m*Tsum);
% h1=hh(1:end)'.*d0; h2=[zeros(1,m) h1(1:(Tsum-1)*m)];
% H=D0*U1; U=U0+D0*U2*D0+H+H';
% vv=v0+d0'.*v1+V2*h2'+V3*h1'+d0'.*(V4*h1');
% LogLike(n)=-1/2*(z'*U*z-z'*vv-vv'*z);


%% mode search iterations
while ~isempty(idx) && isempty(k) && dErr<tol*Err(n)
    % iterate as long as not all constraints are satisfied (idx), the
    % current solution has not already been visited (k), and the change in
    % error (dErr) remains below the tolerance level 
    
    % save last step
    zsv=z; Usv=U; dsv=d;
    if n>1
        if flipAll, dsv(idx)=1-d(idx);
        else dsv(idx(r))=1-d(idx(r)); end;
    end;
    
    % (1) solve for states Z given constraints d
    D=spdiags(d',0,m*Tsum,m*Tsum);
    h1=hh(1:end)'.*d; h2=[zeros(1,m) h1(1:(Tsum-1)*m)];
    H=D*U1; U=U0+D*U2*D+H+H';
    if ~isempty(eps), U=U+eps*speye(size(U)); end;  % avoid singularities
    z=U\(v0+d'.*v1+V2*h2'+V3*h1'+d'.*(V4*h1'));
    
    % (2) flip violated constraint(s)
    idx=find(abs(d-(z>hh)'));
    ae=abs(z(idx)-hh(idx));
    n=n+1; Err(n)=sum(ae); dErr=Err(n)-Err(n-1);
    if flipAll, d(idx)=1-d(idx);    % flip all constraints at once
    else [~,r]=max(ae); d(idx(r))=1-d(idx(r)); end; % flip constraints only one-by-one, 
            % choosing the one with largest violation in each step

    % terminate when revisiting already visited edges:
    l=d*y; k=find(LL==l); LL=[LL l]; 
    
    %disp(n)
    
%     % track log-likelihood (if desired)
%     d0=zeros(1,m*Tsum); d0(z>hh)=1; D0=spdiags(d0',0,m*Tsum,m*Tsum);
%     h1=hh(1:end)'.*d0; h2=[zeros(1,m) h1(1:(Tsum-1)*m)];
%     H=D0*U1; U=U0+D0*U2*D0+H+H';
%     vv=v0+d0'.*v1+V2*h2'+V3*h1'+d0'.*(V4*h1');
%     LogLike(n)=-1/2*(z'*U*z-z'*vv-vv'*z);
    
end;
%if ~isempty(idx), warning('no exact solution found'); end;

if dErr<tol*Err(n)
    % if idx=[] or k!=[], display final error & change in error
    z=reshape(z,m,Tsum);
    if flipAll, d(idx)=1-d(idx);
    else d(idx(r))=1-d(idx(r)); end;
    disp(['dErr: ' num2str(dErr) '  Err: ' num2str(Err(end))])
    Err=Err(2:end);
else
    % if dErr exceeded tolerance, display # of still violated constraints
    z=reshape(zsv,m,Tsum);
    U=Usv;
    d=dsv;
    Err=Err(2:end-1);
    disp(['#viol: ' num2str(length(idx)) '  rev: ' num2str(length(k))])
end;


%%
% (c) 2016 Daniel Durstewitz, Dept. Theoretical Neuroscience, Central
% Institute of Mental Health, Heidelberg University
