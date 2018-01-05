%% illustrates parameter estimation for PLRNN with additional weight matrix C
% for external regressors in Inp
% see Durstewitz (2017), PLoS Comp Biol for further details
%
% z_t = A z_t-1 + W max(z_t-1-h,0) + C Inp_t + e_t , e_t ~ N(0,S)
% x_t = B max(z_t-h,0) + nu_t , nu_t ~ N(0,G)
%
clear all
close all



%% load/set PLRNN_C parameters 
load PLRNNwmParam
M=length(h);    % number of latent states
N=M;    % # of outputs

% create denser input matrix for testing:
s=0; randn('state',s); rand('state',s);
for i=1:2
    InpVal{i}=zeros(10,20);
    r=(rand(10,20)>0.7); InpVal{i}(r)=3-2*i;
    r=(rand(10,20)>0.7); InpVal{i}(r)=2*i-3;
end;

ntr=20; % number of simulated trials
Inp=repmat(InpVal,1,ntr);   % cell array of MxT matrices of external inputs

C=randn(M,10);  % create Mx10 matrix of external regressor weights



%% run perturbed true system for ntr trials
% - generate true but perturbed states Z and observations X
X=cell(1,ntr); Z=cell(1,ntr);
mu0=cell(1,ntr);
for i=1:length(Inp)
    randn('state',i); rand('state',i);
    mu0{i}=randn(M,1);
    [Z{i},X{i}]=SimPLRNN_C(A,W,S,Inp{i},mu0{i},B,G,h,C);
end;
save SimWM_C X Z mu0



%% estimate parameters given true states and observations
% ------------------------------------------------------------------------

% estimate parameters based on true states but estimated (analytically approx.) cov-mtx
Z0=cell2mat(Z);
flipAll=true;   % flip all violated constraints on each iteration 
[Zest,U]=StateEstPLRNN_C(A,W,C,S,Inp,mu0,B,G,h,X,Z0,[],0,[],flipAll);  % estimate Hessian U
            % NOTE: true parameters are used here just to get state estimates right
[EphiziEst,EphizijEst,EziphizjEst,Vest]=ExpValPLRNN(Zest,U,h);  % estimate all other state expectancies 
% estimate parameters based on provided state expectancies:
[mu0_est,B_est,G_est,W_est,A_est,S_est,C_est]=ParEstPLRNN_C(Zest,Vest,EphiziEst,EphizijEst,EziphizjEst,X,Inp);



%% graph results of parameter estimation (estim vs true params)
figure(1), hold off cla

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


figure(2), hold off cla
plot(C(1:end),C(1:end),'k',C(1:end),C_est(1:end),'bo','LineWidth',2);
%axis([-2 3 -2 3]);
set(gca,'FontSize',20); box off; xlabel('true c_i_j'); ylabel('estimated c_i_j');



%%
% (c) 2016 Daniel Durstewitz, Dept. Theoretical Neuroscience, Central
% Institute of Mental Health, Heidelberg University
