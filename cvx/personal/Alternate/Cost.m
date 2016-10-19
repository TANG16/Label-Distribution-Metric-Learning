function cost = Cost(X,R,A)
    cost = 0;
    for i=1:size(X,1)
        for j=i+1:size(X,1)
            if i ~= j
                cost = cost+ R(i,j)*((X(i,:)-X(j,:))*A*(X(i,:)-X(j,:))');
            end
        end
    end
end

    