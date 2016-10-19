clc;
tic
cd('C:\Users\SYARLAG1\Desktop\MatlabLabel Distribution Learning\cvx\personal\Alternate LMNN Approach - diagnol only')
N = csvread('./N.csv');
X = csvread('./X.csv');
m = size(X,2);

cd('../')
cd('../')

cvx_begin
variables A(1,m)
cd('./personal/Alternate LMNN Approach - diagnol only')
minimize((Cost(X,A,N)));
subject to
Const(X,A, N) >= 0; 
A >= 0;
cvx_end
toc