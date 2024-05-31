function [all]=veccomp(ij,n,F);   
for ji=1:n
               all(ji)=(norm(F(ij,:)-F(ji,:)))^2;
              end
