function [PCAVects,m]= DoPCA(Features,dimension)
% Authors: Y.X.Liang, PhD
% Date: Apr. 11, 2010
% Features --- column vector features pool
% dimension --- desired dimension of the resulting PCA subspace
%本身函数：princomp
[dims, nTrainNum] = size(Features);  %Feature 1296*95916
percent = .99;
if(nargin <2)
    dimension=nTrainNum-1;
    percent=.99;%降维后所包含的能量。
end
m = mean(Features,2);  %求平均向量
if(dims<nTrainNum) 
   disp('正在求取特征向量,请稍等.......');
    St = (Features-m*ones(1,nTrainNum)) * (Features-m*ones(1,nTrainNum))'/(nTrainNum-1);
    [V,D]=eig(St);
    [D,idx]=sort(diag(D),'descend')
    d1 = length(find(abs(D)>0.00001));
    d2 = optidim(D,percent);
    d = min([dimension, d1, d2]);
    idx=idx(1:d);
    PCAVects=V(:,idx);
    disp('已经求取了特征向量！');
else
    St =(Features-m*ones(1,nTrainNum))' * (Features-m*ones(1,nTrainNum))/(nTrainNum-1);
    [V,D]=eig(St);
    [D,idx]=sort(diag(D),'descend');
    d1 = length(find(abs(D)>0.00001));
    if(nargin <2)
       d2 = optidim(D,percent);  %这里的percent没有定义
    else
        d2=500;
    end
    d2 = min(d2,500);
    d = min([dimension, d1, d2]) ;
    idx=idx(1:d);
    V = V(:,idx);
    D = D(1:d); 
    D=diag(D.^(-1/2));
    PCAVects = (Features-m*ones(1,nTrainNum))*V*D;
end



function dim = optidim(D,percent)
% Authors: Y.X.Liang, PhD
% Date: Apr. 11, 2010
% Determine the optimal dimension by the distribution of the eigenvalues
% D --- the sorted eigenvalues vector (descend)
if(nargin <2)
    percent = 0.99;
end
total = sum(D);
for dim=1:length(D)
    s=sum(D(1:dim));
    if s/total>=percent,
        break;
    end
end