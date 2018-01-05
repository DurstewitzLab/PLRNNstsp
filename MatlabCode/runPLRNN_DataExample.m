%% illustrates PLRNN state & parameter estimation on multiple single-unit
% recordings from anterior cingulate cortex, originally reported in
% Hyman JM, Whitman J, Emberly E, Woodward TS, Seamans JK. Action and outcome 
%    activity state patterns in the anterior cingulate cortex. 
%    Cereb Cortex. 2013;23: 1257-1268.
%
% for more details see Durstewitz (2017), PLoS Comp Biol
%
%
clear all
close all


%% load data & define inputs
M=10;   % # latent states

load('del_alt_11_26_KDE05');    % spike data filtered with Gaussian kernel & binned at 500 ms,
            % trials in cell array, all trials were cut down to equal time base
k=find(meR>1); SCMall=SCMall(k);    % keep only units with mean rate > 1 Hz
nu=length(SCMall);  % # units
T=size(SCMall{1},2);    % # time bins on each trial
XX=cell2mat(SCMall);
ntr=size(XX,1); % # trials
% shape observations into required format and define inputs (external
% regressors) based on known time points of cues and responses:
X=cell(1,ntr); Inp=cell(1,ntr);
for i=1:ntr
    X{i}=reshape(XX(i,:),T,nu)';
    Inp{i}=zeros(M,T);
    if TrType(i)==2, Inp{i}(1,round(kdt/2)+1:kdt+1)=1;
    else Inp{i}(2,round(kdt/2)+1:kdt+1)=1; end;
    Inp{i}(3,kdt+2*minL:kdt+2*minL+round(kdt/2))=1;
end;


%% set initial parameter estimates
N=size(X{1},1);
randn('state',1); rand('state',1);
mu0_ini=mat2cell(randn(M,length(X)),M,ones(1,length(X)));
B_ini=randn(N,M); G_ini=20*diag(rand(N,1));
W_ini=2*randn(M); W_ini=W_ini-diag(diag(W_ini));
A_ini=diag(rand(M,1));
h=rand(M,1);
S=eye(M);


%% define EM control parameters 
tol=1e-3;   % relative (to first LL value) tolerated increase in log-likelihood
MaxIter=100;    % maximum number of EM iterations allowed
tol2=1e-2;  % relative error tolerance in state estimation (see StateEstPLRNN)
eps=1e-5;   % singularity parameter in StateEstPLRNN
flipOnIt=3; % parameter that controls switch from single (i<=flipOnIt) to 
            % full (i>flipOnIt) constraint flipping in StateEstPLRNN
CtrPar=[tol MaxIter tol2 eps flipOnIt]; % vector of control parameters


%% run EM algo
[mu0_est,B_est,G_est,W_est,A_est,S_est,Ezi,~,~,~,~,LL]= ...
    EMiter(CtrPar,A_ini,W_ini,S,Inp,mu0_ini,B_ini,G_ini,h,X);



%% graph log-likelihood
figure(6), hold off cla
plot(LL,'b','LineWidth',3);
set(gca,'FontSize',20); box off;
xlabel('iteration'); ylabel('log-likelihood');


%%
% (c) 2016 Daniel Durstewitz, Dept. Theoretical Neuroscience, Central
% Institute of Mental Health, Heidelberg University
