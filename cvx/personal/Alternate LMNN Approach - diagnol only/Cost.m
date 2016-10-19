function cost = Cost(X, A, Nei)
    cost = 0;
    for i=1:size(Nei,1)
            for j=1:size(Nei,2)
                if Nei(i,j) > 0
                    cost = cost -Nei(i,j)*(dot((X(i,:)-X(j,:)).^2,A));
                end
            end
    end
end

