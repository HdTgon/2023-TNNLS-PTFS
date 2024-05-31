clc;clear all;

addpath('util');

a='MSRC_v1';

load(a);

[~,v]=size(fea);

[n,~]=size(fea{1});
c=size(unique(gt),1);
class_num=c;
label=gt;
dimension=0;
for num = 1:v
    dimension=dimension+size(fea{num},2);
end
feanum=[10:10:300]; % feature dimension
ACC=[];
NMI=[];
ACCstd=[];
NMIstd=[];
Fscore=[];
Fscorestd=[];
Precision=[];
Precisionstd=[];
Recall=[];
Recallstd=[];
ARI=[];
ARIstd=[];
rankmvufs=[];
mvufsobj=[];
% alpha=[1e-2,1e-1,1,1e1,1e2];
% beta=[1e-2,1e-1,1,1e1,1e2];
% gamma=[1e-2,1e-1,1,1e1,1e2];
% lambda=[1e-2,1e-1,1,1e1,1e2];
alpha=[1e-3,1,1e3];
beta=[1e-5,1e-3,1e-1,1,1e1,1e3,1e5];
gamma=[1e-4,1e-2,1,1e2,1e4];
lambda=[1e4,5e4,1e5];

iter=0;

for i=1:size(alpha,2)
for o=1:size(beta,2)
for s=1:size(gamma,2)
for l=1:size(lambda,2)
 
[P,obj]=Copy_2_of_shrinkPTFS(fea,alpha(i),beta(o),gamma(s),lambda(l),n,c);

iter=iter+1

X=[];
W1 = [];
for num = 1:v
    W1=[W1;sum(P{num}.*P{num},2)];
    X=[X,fea{num}];
end

%% test stage
[~,index] = sort(W1,'descend');
rankmvufs(:,iter)=index;
mvufsobj(1:size(obj,2),iter)=obj;

for j = 1:length(feanum)
acc=[];
nmi=[];
fscore=[];
precision=[];
recall=[];
ari=[];
for k = 1:30
    new_fea = X(:,index(1:feanum(j)));
    idx = kmeans(new_fea, class_num,'MaxIter',200);
    res = bestMap(label,idx);
    acc111 = length(find(label == res))/length(label); % calculate ACC 
    nmi111 = MutualInfo(label,idx); % calculate NMI
    [f111,p111,r111] = compute_f(label,idx);
    [ar111,~,~,~]=RandIndex(label,idx);
    acc=[acc;acc111];
    nmi=[nmi;nmi111];
    fscore=[fscore;f111];
    precision=[precision;p111];
    recall=[recall;r111];
    ari=[ari;ar111];
end
ACC=[ACC;sum(acc)/30];
ACCstd=[ACCstd;std(acc)];
NMI=[NMI;sum(nmi)/30];
NMIstd=[NMIstd;std(nmi)];
Fscore=[Fscore;sum(fscore)/30];
Fscorestd=[Fscorestd;std(fscore)];
Precision=[Precision;sum(precision)/30];
Precisionstd=[Precisionstd;std(precision)];
Recall=[Recall;sum(recall)/30];
Recallstd=[Recallstd;std(recall)];
ARI=[ARI;sum(ari)/30];
ARIstd=[ARIstd;std(ari)];
end

end
end
end
end


number=size(ACC,1);
final=zeros(number,4);
for j=1:number
    final(j,1:12)=[ACC(j,1),ACCstd(j,1),NMI(j,1),NMIstd(j,1),ARI(j,1),ARIstd(j,1),Fscore(j,1),Fscorestd(j,1),Precision(j,1),Precisionstd(j,1),Recall(j,1),Recallstd(j,1)];
end

[newvalue,endindex]=sort(ACC,'descend');

final(endindex(1:10),:);
