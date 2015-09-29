% ========================================================================
%  Block ksvd algorithm
% [D X]=blocksvd(param)
% Inputs
%       param.initdict  -dictionary
%       param.classnum  -class number
%       param.data      -training data 
%       param.subdicnum -number of  atoms for each block
%       param.iternum    -the number of iteration 
%       params.dim;      _the dimension of dictionary
% Outputs
%       D               _dictionary
%       X               -sparsed codes
% Author: Ke Nai
% Date: 3-16-2013
% ========================================================================

function [D,X]= dblocksvd(params,varargin)  
   c=params.classnum ;           
   cnum=params.subdicnum ;       
   iter=params.iternum;          
   tdata=params.data ;          
   dict=params.initdict;         
   dimen=params.dim;   
   label=params.label;
   dictnum=c*cnum;             
   trainnum=size(tdata,2);     
   X=zeros(dictnum,trainnum);
   
   
     res3=zeros(1,iter);
     
   for i=1:iter   
       i
       X=lcsbomp(params);                       %using sbomp to perform sparse coding
       ind=1:cnum;                            %the indics for each block
       for j=1:c
           if j~=1
              ind=ind+cnum;     
           end
           index=find(sum(X(ind,:).^2)) ;       %find the traing samples which use the jth block
           index2=find(label(j,:)==1)  ;     %the index of training samples correspond to class j
           X2=X(ind,index);
           if length(index)<1                   %there is no training sample use the jth block
               ErrorMat=tdata-dict*X;           %using the error of all training samples to update 
               ErrorNormVec = sum(ErrorMat.^2);
               [d,p] = maxn(ErrorNormVec,cnum);
               betterDictionaryElement =tdata(:,p);
               betterDictionaryElement = betterDictionaryElement./repmat(sqrt(sum( betterDictionaryElement.^2)),size(betterDictionaryElement,1),1);
               dict(:,ind) = betterDictionaryElement;
               X(ind,:) = 0;
           else                       
               tempx=X(:,index);                 %puck up the training samples that use the jth block  
               tempx(ind,:)=0;                   %set the coefficient to zeros
               errors=tdata(:,index)-dict*tempx;  %obtain the error of training sample that discard the jth block
             
               id2=ismember(index,index2) ; 
               
               if ~any(id2)
                  1
                 [uu,ss,vv]=svd(tdata(:,index2));     
                  pp=1:cnum;  
                  dict(:,ind) = uu(:,pp);          %simultaneously the block
                  
                  coeff=zeros(size(X2,1),size(X2,2));
                   for k=1:length(index)
                        xx=dict(:,ind)\errors(:,k);
                        coeff(:,k)=xx;
                   end         
               else
                   [uu,ss,vv]=svd(errors(:,id2));              
                   summ=min(size(uu,1),size(vv,1));
                   tempu=uu(:,1:summ);
                   temps=ss(1:summ,1:summ);
                   tempv=vv(:,1:summ);
                   pp=1:cnum;  
                   dict(:,ind) = tempu(:,pp);          %simultaneously the block 
                   diagele=diag(temps);
                   coeff2=getcoeff(diagele,tempv,pp);   %simultaneously sparse codes          
                   coeff=zeros(size(X2,1),size(X2,2));
                   coeff(:,id2)=coeff2';
                   cc=ones(1,length(index));
                   index2=logical(cc-id2);
                   xx=dict(:,ind)\errors(:,index2);
                   coeff(:,index2)=xx;
             end
               X(ind,index)=coeff; 
               
           end   
       end
       params.initdict=normcols(dict);          
       res3(i)=norm(tdata-dict*X,'fro');          
   end
 D=normcols(dict);    
end


function [d,p]=maxn(vec,cnum)   
    for i=1:cnum
        [val pos]=max(vec);
        d(i)=val;
        p(i)=pos;
        vec(pos)=0;
    end
end


function [d,p]=minn(vec,cnum) 
    for i=1:cnum
        [val pos]=min(vec);
        d(i)=val;
        p(i)=pos;
        vec(pos)=inf;
    end
end


function coeff=getcoeff(diagele,tempv,pp) 
    l=length(pp);
    for i=1:l
        coeff(:,i)=diagele(pp(i))*tempv(:,pp(i));
    end
end

