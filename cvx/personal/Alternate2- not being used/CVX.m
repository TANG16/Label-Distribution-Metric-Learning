clear;
cd('C:\Users\SYARLAG1\Desktop\Label Distribution Learning\cvx\personal\Alternate2')
X = csvread('./X.csv');
R = csvread('./R.csv');
S = csvread('./S.csv');
D = csvread('./D.csv');


m = size(X,2);

cd('../')
cd('../')

cvx_begin
variables A(m,m)
cd('./personal/Alternate2')
minimize(Cost(X,S,D,A));
subject to
A >= 0;
cvx_end