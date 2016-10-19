function cost = XingConst(X,D,A)
    cost = 0;
    for i=1:size(X,2)
        for j=1:size(X,2)
            if i ~= j
                if D(i,j) == 1
                    cost = ((X(i,:)-X(j,:))*A*(X(i,:)-X(j,:))')^0.5;
                end
            end
        end
    end
end

    