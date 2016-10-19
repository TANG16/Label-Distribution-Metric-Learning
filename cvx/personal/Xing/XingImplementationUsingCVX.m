cd('C:\Users\SYARLAG1\Desktop\Label Distribution Learning\cvx\personal')
X = csvread('./data.csv');
S = csvread('S.csv');
D = csvread('D.csv');

m = size(X,2);

cd('../')

cvx_begin
variables A(m,m)
cd('./personal')
minimize(XingCost(X,S,A));
subject to 
XingConst(X,D,A) >= 1;
A >= 0;
cvx_end