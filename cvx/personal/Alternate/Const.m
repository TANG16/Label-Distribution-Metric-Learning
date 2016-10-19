function cost = Const(X,S,R,A)
    cost = 0;
    for i=1:size(X,2)
        for j=1:size(X,2)
            if i ~= j
                if S(i,j) < .10
                    cost = sqrt((X(i,:)-X(j,:))*A*(X(i,:)-X(j,:))');
                end
            end
        end
    end
end

    