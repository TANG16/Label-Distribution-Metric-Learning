function cost = XingCost(X,S,A)
    cost = 0;
    for i=1:size(X,2)
        for j=1:size(X,2)
            if i ~= j
                if S(i,j) == 1
                    cost = (X(i,:)-X(j,:))*A*(X(i,:)-X(j,:))';
                end
            end
        end
    end
end

    