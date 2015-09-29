% ========================================================================
% update dictionary atoms
% USAGE: [D X] =DictionaryUpdate(Y,Dinit,H_train,iterations,sparsitythres,dictsize)
% Inputs
%       Y               -training features
%       Dinit           -initialized dictionary
%       H_train         -label matrix for training feature 
%       dictsize        -number of dictionary items
%       iterations      -iterations
%       sparsitythres   -sparsity threshold
% Outputs
%       D               -learned dictionary
%       X               -sparsed codes

 %Author: Ke Nai
% ========================================================================
function [D1 X1]=DictionaryUpdate(Y,Dinit,H_train,iterations,sparsitythres,dictsize,testing_feats,H_test)
    params.classnum = size(H_train,1);   
    params.subdicnum = round(dictsize/size(H_train,1)); 
    params.data =Y;                      
    params.Tdata = sparsitythres; 
    params.iternum = iterations;          
    D_ext2 = [Dinit];                      
    D_ext2=normcols(D_ext2); 
    params.initdict = D_ext2;
    params.dim=size(Dinit,1);
    params.label=H_train;
    
    params.data2=testing_feats;
    params.tlabel=H_test;

    [D1,X1] = dblocksvd(params,'');  
end
