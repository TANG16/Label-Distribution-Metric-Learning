function [const] = Const(X, A, Nei)
    const = [];
    for i =1:size(Nei,1)
        for j1 =1:size(Nei,2)
            if Nei(i,j1) > 0
               for j2=1:size(Nei,2)
                   if Nei(i,j2) > 0
                       const = [const (Nei(i,j2)*(dot((X(i,:)-X(j2,:)).^2,A))+Nei(i,j1)*(dot((X(i,:)-X(j1,:)).^2,A)))];
                   end
               end
            end
        end
    end
end

