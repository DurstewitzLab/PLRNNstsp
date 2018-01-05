function [Ephizi,Ephizij,Eziphizj,V]=ExpValPLRNN(Ez,U,h)
%
% please cite (& consult for further details): Durstewitz (2017) PLoS Comp Biol
% 
% computes expectancies E[phi(z)], E[z phi(z)'], E[phi(z) phi(z)'],  
% as given in eqn. 10-15, based on provided state expectancies and Hessian
% 
% REQUIRED INPUTS:
% Ez: MxT matrix of state expectancies as returned by StateEstPLRNN
% U: negative Hessian of log-likelihood returned by StateEstPLRNN 
% h: Mx1 vector of thresholds
%
% OUTPUTS:
% Ephizi: E[phi(z)]
% Ephizij: E[phi(z) phi(z)']
% Eziphizj: E[z phi(z)']
% V: state covariance matrix E[zz']-E[z]E[z]'

eps=1e-4;   % minimum enforced variance/ eigenvalue

[m,T]=size(Ez);

%% invert block-tridiagonal neg. Hessian U
U0=zeros(m*T,2*m);
for t=1:T
    k0=(t-1)*m+1:t*m;
    k1=max(0,(t-2)*m)+1:t*m;
    U0(k0,1:2*m)=[zeros(m,(t<2)*m) U(k0,k1)];
end;
V0=invblocktridiag(U0,m);
v=V0(m+1:end,1:m)';
for t=1:T-1, V0((t-1)*m+1:t*m,2*m+1:3*m)=v(1:m,(t-1)*m+1:t*m); end;
k0=(1:T*m)'*ones(1,3*m);
k1=ones(T*m,1)*(1:3*m);
k1(2*m+1:end,:)=k1(2*m+1:end,:)+reshape(repmat(m:m:(T-2)*m,m,3*m),(T-2)*m,3*m);
k1((T-1)*m+1:T*m,:)=k1((T-2)*m+1:(T-1)*m,:);
V0(1:m,:)=circshift(V0(1:m,:),-m,2);
V0((T-1)*m+1:T*m,:)=circshift(V0((T-1)*m+1:T*m,:),m,2);
V=sparse(k0,k1,V0);

% ensure proper covariance matrix
V=(V+V')./2;    % ensure symmetry
for i=1:m*T, V(i,i)=max(V(i,i),eps); end;   % restrict min var


Ez=Ez(1:end)';
hh=repmat(h,T,1);
v=spdiags(V,0);
s=sqrt(v);
fti=normpdf(hh,Ez,s+zeros(size(hh)));
Fti=normcdf(hh,Ez,s+zeros(size(hh)));
a0=v.*fti;
Exi=a0+Ez.*(1-Fti);
Exi2=(Ez.^2+v).*(1-Fti)+(hh+Ez).*a0;


%% E[phi(zi)]
a1=hh.*(1-Fti);
Ephizi=Exi-a1;


%% E[phi(zi)*phi(zj)]
o1=ones(m*2,1);
J1=sparse(m*T,m*T); J2=sparse(m*T,m*T);
for t=1:T-1
    k0=(t-1)*m+1:(t+1)*m;
    k1=t*m+1:(t+2)*m;
    mu=Ez(k0)*o1';
    
    % 1) compute all required lam_j and lam_ji elements
    Q=v(k0)*v(k0)'-V(k0,k0).*V(k0,k0);
    Ljj=(o1*v(k0)')./Q;
    Lji=-V(k0,k0)./Q;

    % 2) compute sigmas
    SI0=Lji.^2.*(o1*v(k0)')+Ljj;
    SIG=1./(1./(o1*v(k0)')+Lji.^2./Ljj);
    Y=(Ljj.*(mu-hh(k0)*o1')).^2./SI0;
    rho=sqrt(SIG./(2*pi*Ljj.*(o1*v(k0)'))).*exp(-1/2*Y);

    % 3) compute mu's
    MU=mu'+(Lji.*Ljj.*(o1*v(k0)').*(mu-hh(k0)*o1'))./SI0;

    % 4) compute first 3 terms of eq. 12
    fphiti=normpdf(hh(k0)*o1',MU',sqrt(SIG'));
    Fphiti=normcdf(hh(k0)*o1',MU',sqrt(SIG'));
    PI=rho'.*(SIG'.*fphiti+MU'.*(1-Fphiti));
    PII=(mu'.*(Exi(k0)*o1')+(Lji./Ljj)'.*(mu.*(Exi(k0)*o1')-(Exi2(k0)*o1'))).* ...
        (1-normcdf(o1*hh(k0)',mu',sqrt(1./Ljj)'));
    J1(k0,k0)=PI+PII;

    PI=rho'.*(1-Fphiti);
    PII=((mu'+(Lji./Ljj)'.*mu).*(1-Fti(k0)*o1')-(Lji./Ljj)'.*(Exi(k0)*o1')).* ...
        (1-normcdf(o1*hh(k0)',mu',sqrt(1./Ljj)'));
    J2(k0,k0)=(PI+PII).*(hh(k0)*o1');
    
    if t<T-1, J1(k1,k1)=0; J2(k1,k1)=0; end;
end;

% 5) move along block-diagonal and compute 4th term of eq. 12
J3=sparse(m*T,m*T);
for i=1:T*m-1
    if i>(T-1)*m, k=m-mod(i-1,m); else k=2*m-mod(i-1,m); end;
    for j=2:k
        V1=V([i i+j-1],[i i+j-1]);
        
        if ~isempty(find(V1-V1'))
            warning(['V1 not sym.: ' num2str(V1(1:end))]);
        end;
        
        if min(spdiags(V1,0))<eps
            warning(['V1 with var<eps: ' num2str(V1(1:end))]);
        end;
        
        [U,E]=eigs(V1); E=diag(E);
        if min(E)<eps  % ensure positive-definiteness
            %warning(['V1 not pos.-def.: ' num2str(E')]);
            E=max(E,max(eps,1e-8*max(E))); V1=U'*diag(E)*U; V1=(V1+V1')./2;
        end;
        
        J3(i,i+j-1)=mvncdf(-[hh(i) hh(i+j-1)],-[Ez(i) Ez(i+j-1)],V1);
        if isnan(J3(i,i+j-1))
            J3(i,i+j-1)=mvncdf(-[hh(i) hh(i+j-1)],-[Ez(i) Ez(i+j-1)],diag(diag(V1)));
        end;
        J3(i+j-1,i)=J3(i,i+j-1);
        ij=[i i+j-1];
        J3(ij,ij)=(hh(ij)*hh(ij)').*J3(ij,ij);
    end;
end;

% 6) combine all this into eq. 12
Ephizij=J1-J2-J2'+J3;


%% E[phi(zi)*phi(zi)]
Ephizij=triu(Ephizij,1)+tril(Ephizij,-1)+spdiags(Exi2-2*hh.*Exi+hh.*a1,0,m*T,m*T);


%% E[zi*phi(zj)]
Eziphizj=sparse(m*T,m*T);
for t=1:T-1
    k0=(t-1)*m+1:(t+1)*m;
    L=V(k0,k0)./(o1*v(k0)'); % -vij/vjj=lam_i^-1*lam_ij;
    mu=Ez(k0)*o1';
    Eziphizj(k0,k0)=L.*(o1*Exi2(k0)')+(mu-L.*(mu'+o1*hh(k0)')).*(o1*Exi(k0)')- ...
        (o1*hh(k0)').*(mu-L.*mu').*(1-o1*Fti(k0)');
end;


%%
% (c) 2016 Daniel Durstewitz, Dept. Theoretical Neuroscience, Central
% Institute of Mental Health, Heidelberg University
