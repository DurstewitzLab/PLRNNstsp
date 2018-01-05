function q=ExtractBlockDiag(C,M)
% extracts block-tridiagonal elements from matrix C with M-1 elements on
% each side from main diagonal
k=-(M-1):(M-1);
Q=spdiags(C,k);
kk=zeros(length(Q),1); kk(1:M:end)=1;
q=[];
for i=1:M-1
    q=[q Q(find(kk),i)'];
    kk(i+1:M:end)=1;
end;

%%
% (c) 2016 Daniel Durstewitz, Dept. Theoretical Neuroscience, Central
% Institute of Mental Health, Heidelberg University
