function x=gomp(param)
%=============================================
% Group Sparse coding of a group of signals based on a given 
% dictionary and specified number of atoms to use. 
% input arguments: A - the dictionary
%                  y - the signals to represent
%                  group - group indices, should be a vector with dimensionality of col of A
%                  err - the maximal allowed representation error for
%                  each siganl.
% output arguments: x - sparse coefficient .
%Author: Ke Nai
%===============================================

dic=param.initdict;
traindata=param.data;
x = zeros(size(dic,2),size(traindata,2));
sparsity=param.Tdata ;
cnum=param.subdicnum;
g = cell(param.classnum,1);
for i=1:size(traindata,2)
    
    for j = 1:param.classnum 
       g{j} =(j-1)*cnum+[1:cnum];
    end
    r=traindata(:,i);
    L=[];   
    p=1;
    while p<=sparsity
%         fprintf('\n正在进行第%d次迭代！\n',p);
          l=dic'*r; 
          lg = zeros(param.classnum ,1);
          for j = 1:param.classnum 
                lg(j) = norm(abs(l(g{j})));
           
          end
          [temp, idx] = sort(lg, 'descend');
          L=[L  g{idx(1)}];
          Psi = dic(:,L);
          x_bar =pinv(Psi)*traindata(:,i);
          p=p+1;
          r = traindata(:,i) - Psi*x_bar;
%           fprintf('\n本次残差：%f',norm(r));
%            fprintf('\n');
    end
    x(L,i)=x_bar;
end
end