function cost = Cost(X,S,D,A)
    cost = 0;
    for i=1:size(X,2)
        for j=1:size(X,2)
            if i ~= j
                cost = cost - S(i,j)*((X(i,:)-X(j,:))*A*(X(i,:)-X(j,:))') + D(i,j)*((X(i,:)-X(j,:))*A*(X(i,:)-X(j,:))');
            end
        end
    end
end

    