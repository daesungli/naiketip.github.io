  clear all;
  clc;
  addpath(genpath('.\ksvdbox'));  % add K-SVD box
  addpath(genpath('.\large_scale_svm'));  % add K-SVD box
  addpath(genpath('.\OMPbox')); % add sparse coding algorithem OMP
 

%% constant
   sparsitythres=5; %block sparsity prior 
   sparsitythres2=[5 8 10 12 15 18]; %sparsity prior for initialization 
   dictsize=[500 800  1000  1200  1500  1800];  % dictionary size        
   iterations=10; % iteration number                  
   iterations4ini=20; % iteration number for initialization  
   
   train=dir('.\trainingdata\AR');
   
   str='.\trainingdata\AR\';
     
  accuracy1=0;
  
  database='AR';
  
  
  res=[];
  for j=1:1
      
   for i=3:3
     i
     str2=[str train(i).name]
     
     load(str2);
     
     numofatom=dictsize(j)/size(H_train,1);
     
     ind1=findstr(train(i).name,'.');
     
     ind2=findstr(train(i).name,'R');
     
     num=train(i).name(ind2(end)+1:ind1-1);
     
     
   %% dictionary initialization   
   fprintf('\n Block k-svd initialization... ');
      
    t1 = clock; 
   [Dinit] = initializationbksvd(training_feats,H_train,dictsize(j),iterations4ini,sparsitythres2(j));
    fprintf('done!');
   
   %% dictionary learning process
   fprintf('\nDictionary learning by Block k-svd...');
   [D1 X1] = DictionaryUpdate(training_feats,Dinit,H_train,iterations,sparsitythres,dictsize(j),testing_feats,H_test);  %正式学习字典
 
    etime(clock,t1);
     
    time=ans;
    
     str2=['.\DBKSVD-AR-' num2str(sparsitythres) '-' num2str(numofatom)];
           
     save(str2,'time');
        
     fprintf('\nClassification...');  
     [prediction,accuracy,count,x]=classification(D1,testing_feats,H_test,sparsitythres); %分类测试样本

    res=[res accuracy];
      
      
   end
  end


