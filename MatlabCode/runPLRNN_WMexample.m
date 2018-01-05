%% illustrates PLRNN state & parameter estimation, and full EM, for working 
% memory example from Durstewitz (2017), PLoS Comp Biol (please see this
% ref. for further details); generates Figs. 3 & 4
%
%
clear all
close all


%% set parameters of simulated (ground truth) system
load PLRNNwmParam
M=length(h);    % number of latent states
ntr=20; % number of simulated trials
Inp=repmat(InpVal,1,ntr);   % cell array of MxT matrices of external inputs
N=M;    % # of outputs


%% run perturbed true system for ntr trials
% - generate true but perturbed states Z and observations X
for i=1:length(Inp)
    randn('state',i); rand('state',i);
    mu0{i}=randn(M,1);
    [Z{i},X{i}]=SimPLRNN(A,W,S,Inp{i},mu0{i},B,G,h);
end;
save SimWM X Z mu0



%% infer states given true parameters and observations
% ------------------------------------------------------------------------
flipAll=true;   % flip all violated constraints on each iteration 
Zest0=StateEstPLRNN(A,W,S,Inp,mu0,B,G,h,X,[],[],[],[],flipAll);


%% graph results of state estimation
% - plot true & estimated states for one trial of each type
figure(3), hold off cla
L=cell2mat(cellfun(@size,X,'UniformOutput',false)');
[nu,TrialLength]=size(X{1});
Zest=mat2cell(Zest0,nu,L(:,2)');
tr0=4;  % pick some trial for example
for i=1:2
    subplot(1,3,i+1), hold off cla
    plot(1:TrialLength,Z{tr0+i}(3,:),'bo-',1:TrialLength,Zest{tr0+i}(3,:),'b*--', ...
        1:TrialLength,Z{tr0+i}(4,:),'ro-',1:TrialLength,Zest{tr0+i}(4,:),'r*--', ...
        'LineWidth',2,'MarkerSize',10);
    set(gca,'FontSize',20); box off; title(['Input unit #' num2str(2*i-1)]);
    xlabel('time step'); ylabel('activation');
    if i==1
        legend('true','estim','Location','NorthEast','Orientation','horizontal');
        legend('boxoff');
    end;
    yy=-2:0.1:3;
    hold on, plot(zeros(1,length(yy))+5,yy,'k--','LineWidth',2);
    axis([0 TrialLength+1 -2 3])
    title(['Trial type #' num2str(i)])
end;

% - plot overall correlation between estimated and true states
subplot(1,3,1), hold off cla
Z0=cell2mat(Z);
plot(Z0(1:end),Zest0(1:end),'r.','LineWidth',3);
hold on, plot(Z0(1:end),Z0(1:end),'k-','LineWidth',2);
axis([-5 5 -5 5]);
set(gca,'FontSize',20); box off;
xlabel('true states'); ylabel('estimated states');
[r,p]=corr(Z0(1:end)',Zest0(1:end)');
text(-4,3,['r \approx ' num2str(round(r*1000)/1000)],'FontSize',20);





%% estimate parameters given true states and observations
% ------------------------------------------------------------------------
clear all
load PLRNNwmParam
ntr=20; % number of simulated trials
Inp=repmat(InpVal,1,ntr);
load SimWM

% estimate parameters based on true states but estimated (analytically approx.) cov-mtx
Z0=cell2mat(Z);
flipAll=true;   % flip all violated constraints on each iteration 
[Zest,U]=StateEstPLRNN(A,W,S,Inp,mu0,B,G,h,X,Z0,[],0,[],flipAll);  % estimate Hessian U
[EphiziEst,EphizijEst,EziphizjEst,Vest]=ExpValPLRNN(Zest,U,h);  % estimate all other state expectancies 
[mu0_est,B_est,G_est,W_est,A_est,S_est]=ParEstPLRNN(Zest,Vest,EphiziEst,EphizijEst,EziphizjEst,X,Inp); % estimate parameters


%% graph results of parameter estimation (estim vs true params)
figure(4), hold off cla
mu0_=cell2mat(mu0); mu0_=mu0_(1:end);
mu0_est_=cell2mat(mu0_est); mu0_est_=mu0_est_(1:end);
subplot(2,3,1), hold off cla, plot(mu0_,mu0_,'k-','LineWidth',2); hold on
plot(mu0_,mu0_est_,'b.','LineWidth',3,'MarkerSize',15);
axis([-3 3 -3 3]);
set(gca,'FontSize',20); box off; xlabel('true \mu_0'); ylabel('estimated \mu_0');

subplot(2,3,2), hold off cla
bar([diag(A)';diag(A_est)']');
set(gca,'FontSize',20); box off;
set(gca,'XTick',1:5,'XTickLabel',{'a_1_1','a_2_2','a_3_3','a_4_4','a_5_5'});
legend('true','estim'); legend('boxoff');
axis([0.5 5.5 0 0.6])

subplot(2,3,3), hold off cla
k=find(W~=0);
plot(W(k),W(k),'k',W(k),W_est(k),'bo','LineWidth',2);
axis([-1 1.5 -1 1.5]);
set(gca,'FontSize',20); box off; xlabel('true w_i_j'); ylabel('estimated w_i_j');

subplot(2,3,4), hold off cla, bar([diag(S)';diag(S_est)']');
set(gca,'FontSize',20); box off;
set(gca,'XTick',1:5,'XTickLabel',{'\sigma^2_1_1','\sigma^2_2_2','\sigma^2_3_3','\sigma^2_4_4','\sigma^2_5_5'});
axis([0.5 5.5 0 0.02])

subplot(2,3,5), hold off cla
plot(B(1:end),B(1:end),'k',B(1:end),B_est(1:end),'bo','LineWidth',2);
axis([-2 3 -2 3]);
set(gca,'FontSize',20); box off; xlabel('true b_i_j'); ylabel('estimated b_i_j');

subplot(2,3,6), hold off cla
bar([diag(G)';diag(G_est)']');
set(gca,'FontSize',20); box off;
set(gca,'XTick',1:5,'XTickLabel',{'\gamma^2_1_1','\gamma^2_2_2','\gamma^2_3_3','\gamma^2_4_4','\gamma^2_5_5'});
axis([0.5 5.5 0 0.03])





%% run full EM algo on WM example
% ------------------------------------------------------------------------
clear all

% generate example data with 20 observations
load PLRNNwmParam
randn('state',0); rand('state',0);
M=length(h);    % number of latent states
ntr=20; % number of simulated trials
Inp=repmat(InpVal,1,ntr);   % cell array of MxT matrices of external inputs
N=20;    % # of outputs
B=randn(N,M);   % NxM matrix of regression weights
G=diag(1e-2*ones(N,1)); % NxN diagonal observation noise covariance matrix
for i=1:length(Inp)
    randn('state',i); rand('state',i);
    mu0{i}=randn(M,1);
    [Z{i},X{i}]=SimPLRNN(A,W,S,Inp{i},mu0{i},B,G,h);
end;
save SimWM2 X Z mu0

% --- initial parameter estimates 
B_ini=randn(N,M); G_ini=diag(rand(N,1));
W_ini=randn(M); W_ini=W_ini-diag(diag(W_ini));
A_ini=diag(rand(M,1));
mu0_ini=mat2cell(randn(M,length(X)),M,ones(1,length(X)));

% run EM algorithm
tol=1e-3;   % relative (to first LL value) tolerated increase in log-likelihood
MaxIter=100;    % maximum number of EM iterations allowed
tol2=1e-2;  % relative error tolerance in state estimation (see StateEstPLRNN)
eps=1e-5;   % singularity parameter in StateEstPLRNN
flipOnIt=5; % parameter that controls switch from single (i<=flipOnIt) to 
            % full (i>flipOnIt) constraint flipping in StateEstPLRNN
CtrPar=[tol MaxIter tol2 eps flipOnIt]; % vector of control parameters
[mu0_est,B_est,G_est,W_est,A_est,S_est,Ez,Vest,Ephizi,Ephizij,Eziphizj,LL]= ...
    EMiter(CtrPar,A_ini,W_ini,S,Inp,mu0_ini,B_ini,G_ini,h,X);

% NOTE: for mapping estimated onto true parameters, states may have to be
% re-ordered to align with true states, i.e. columns/rows of matrices B, A,
% W, S may have to be swapped!!!

%% graph log-likelihood
figure(5), hold off cla
plot(LL,'LineWidth',3);
set(gca,'FontSize',20); box off;
xlabel('iteration'); ylabel('log-likelihood');



%%
% (c) 2016 Daniel Durstewitz, Dept. Theoretical Neuroscience, Central
% Institute of Mental Health, Heidelberg University
