function [Pv,obj]=Copy_2_of_shrinkPTFS(fea,alpha,beta,gamma,lambda,n,c)
[~,v]=size(fea);
lossv=cell(1,v);
weight=cell(1,v);
averageloss=zeros(v,1);
stdloss=zeros(v,1);
threshold=zeros(v,1);
Fv=cell(1,v);
Av=cell(1,v);
Pv=cell(1,v);
Lv=cell(1,v);
d=zeros(v,1);
Dv=cell(1,v);
% etav=zeros(v,1);
Sv=cell(1,v);
Qv=cell(1,v);
Jv=cell(1,v);
tau=1e3;
pho=0.1; 
mu=2;
maxiter=30;
sT = [n, n, v];

for num=1:v
    weight{num}=zeros(n,1);
    d(num)=size(fea{num},2);
    S=constructW_PKN(fea{num}');
    Sv{num}=(S+S')/2;
    Lv{num}=diag(sum(S))-S;
    [Fv{num}, ~, ~]=eig1(Lv{num}, c, 0);
    Pv{num}=randn(d(num),c);
    Av{num}=Fv{num}*Fv{num}';
    Dv{num}=eye(d(num));
    Jv{num}=zeros(n,n);
    Qv{num}=zeros(n,n);
end

    for iter=1:maxiter
        %%loss vector and threshold
        for num=1:v
        weight{num}=zeros(n,1);
        lossv{num}=sum((Fv{num}-fea{num}*Pv{num}).*(Fv{num}-fea{num}*Pv{num}),2);
        averageloss(num)=sum(lossv{num})/n;
        stdloss(num)=std(lossv{num});
        threshold(num)=averageloss(num)+iter*stdloss(num)/maxiter;
        weight{num}((lossv{num}<threshold(num)))=1;
        end
        
        %%update P
        for num=1:v
            U=diag(weight{num})^(1/2);
            G=U*fea{num};Q=U*Fv{num};
            Pv{num}=(G'*G+gamma*Dv{num})\(G'*Q);
            Pi=sqrt(sum(Pv{num}.*Pv{num},2)+eps);
            diagonal=0.5./Pi;
            Dv{num}=diag(diagonal);
        end
        
        %%update S             
        for num = 1:v
            temp_S=zeros(n);
            distX = L2_distance_1(fea{num}',fea{num}');
            distF = L2_distance_1(Fv{num}',Fv{num}');
            for j = 1:n  
                ad = (distX(j,:)+1/2*beta*distF(j,:))/(2*alpha);
                temp_S(j,:) = EProjSimplex_new(-ad);
            end
            Sv{num} = temp_S;
            Lv{num} = diag(sum(Sv{num}+Sv{num}'))-(Sv{num}+Sv{num}')/2;
        end
        
        %update Y
        for num=1:v
            U=diag(weight{num})^(1/2);
            AAA=Qv{num}+Qv{num}'+2*beta*Lv{num}-pho*Jv{num}-pho*Jv{num};
            A_plus = (abs(AAA)+AAA)/2;
            A_minus = (abs(AAA)-AAA)/2;
            BBB=2*U'*U*fea{num}*Pv{num};
            B_plus = (abs(BBB)+BBB)/2;
            B_minus = (abs(BBB)-BBB)/2;
            term_1=A_minus*Fv{num}+B_plus+4*tau*Fv{num}+eps;
            term_2=A_plus*Fv{num}+B_minus+(2*pho+4*tau)*Fv{num}*Fv{num}'*Fv{num}+2*U'*U*Fv{num}+eps;
            Fv{num} = Fv{num}.*sqrt(term_1./term_2);
            Fv{num} = Fv{num}*diag(sqrt(1./(diag(Fv{num}'*Fv{num})+eps)));
        end

%         %%update nv
%         for num = 1:v
%             etav(num) = 0.5/((trace(Fv{num}'*Lv{num}*Fv{num}))^(1/2));
%         end
             
        %%update J
        for num=1:v
            Av{num}=Fv{num}*Fv{num}';
        end
        A_tensor = cat(3, Av{:,:});
        Q_tensor = cat(3, Qv{:,:});
        a = A_tensor(:);
        q = Q_tensor(:);
        [j, ~] = wshrinkObj(a + 1 / pho * q, lambda/ pho, sT, 0, 3);
        J_tensor = reshape(j, sT);
        for num=1:v
            Jv{num} = J_tensor(:,:,num);      
        end
        
        %%update multipliers
        for num=1:v
            Qv{num}=Qv{num}+pho*(Av{num}-Jv{num});  
        end
        pho=pho*mu;

        %%obj
        leq = A_tensor-J_tensor; 
        leqm = max(abs(leq(:)));
        err = max([leqm]);
        obj(iter)=err;
        if iter>2
            if err < 1e-9
                break;
            end
        end
        
    end

end


