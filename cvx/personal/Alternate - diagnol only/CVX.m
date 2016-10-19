clear;
tic
cd('C:\Users\SYARLAG1\Desktop\Label Distribution Learning\cvx\personal\Alternate')
X = csvread('./X.csv');
R = csvread('./R.csv');
S = csvread('./S.csv');


m = size(X,2);

cd('../')
cd('../')

cvx_begin
variables A(1,m)
cd('./personal/Alternate - diagnol only')
minimize((Cost(X,R,A)));
%minimize((Cost(X,R,A))+ norm_nuc(A));
subject to
%Const(X,S,R,A) >= 1 %the margin?
%sum(A) = 1;
A >= 0;
cvx_end
toc