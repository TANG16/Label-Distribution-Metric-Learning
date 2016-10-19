%if first time running, run cvx_setup in cvx folder, before running this.
clear;
tic
cd('C:\Users\SYARLAG1\Desktop\MatlabLabel Distribution Learning\cvx\personal\Alternate')
X = csvread('./X.csv');
R = csvread('./R.csv');
%S = csvread('./S.csv');


m = size(X,2);

cd('../')
cd('../')

cvx_begin
variables A(m,m)
cd('./personal/Alternate')
minimize((Cost(X,R,A))-log_det(A));
%minimize((Cost(X,R,A))+ norm_nuc(A));
subject to
%Const(X,S,R,A) >= 1 %the margin?
A >= 0;
cvx_end
toc