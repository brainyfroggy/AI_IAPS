clear

mainpath = 'F:\yujun\projects\AI_IAPS\GLM_singletrial\betas';
mask=spm_vol('F:\yujun\projects\data\masks\kastner79.nii');
mask=spm_read_vols(mask);

for Permu=1:10
    
    for MaskNum=1:25

        mask_f=find(mask==MaskNum);
        if MaskNum==18
          mask_f=[find(mask==MaskNum);find(mask==MaskNum+1);find(mask==MaskNum+2);find(mask==MaskNum+3);find(mask==MaskNum)+4;find(mask==MaskNum+5)];
        end
        mask_r=mask_f; 
        
        for SubNum=1:7
            
            %%%%%%%%%% Configure%%%%%%%%%%%
            Nt = processNIfTIData(mainpath, SubNum, '\neutral');            
            Pl = processNIfTIData(mainpath, SubNum ,'\pleasant');
            Up = processNIfTIData(mainpath, SubNum ,'\unpleasant');
            NtAI = processNIfTIData(mainpath, SubNum ,'\neutralAI');
            PlAI = processNIfTIData(mainpath, SubNum ,'\pleasantAI');
            UpAI = processNIfTIData(mainpath, SubNum ,'\unpleasantAI');
      
            Ntm=Nt(mask_r,:);Ntr=Ntm(isnan(Ntm(:,1))==0,:);Ntr=Ntr';
            Upm=Up(mask_r,:);Upr=Upm(isnan(Upm(:,1))==0,:);Upr=Upr';
            Plm=Pl(mask_r,:);Plr=Plm(isnan(Plm(:,1))==0,:);Plr=Plr';
%             Ntm=Nt(mask_r,:);Ntr=Ntm(isnan(Ntm(:,1))==0,:);Ntr=Ntr';
%             Upm=Up(mask_r,:);Upr=Upm(isnan(Upm(:,1))==0,:);Upr=Upr';
%             Plm=Pl(mask_r,:);Plr=Plm(isnan(Plm(:,1))==0,:);Plr=Plr';
            
            CombinedData=cat(1,Plr,Ntr);
            ZScoredData=zscore(CombinedData,0,2);

            Labels=zeros(size(CombinedData,1),1);% Labels Left is zeros
            Labels((size(Ntr,1)+1):end,1)=1;
            Labels(1:100)=-1*ones(100,1);
            
            %cross validation%
            Indices = crossvalind('Kfold', Labels,10); 
            for i=1:10
                test=Indices==i;
                train=~test;

                SVMModel=fitcsvm(double(ZScoredData(train,:)),Labels(train) ) ;
                [computedLabels, score] = predict(SVMModel, double(ZScoredData(test,:)));
                % Confusion Matrix
                [cMatrix, cOrder] = confusionmat(Labels(test), computedLabels);
                bias = SVMModel.Bias;
                w = SVMModel.Beta;  
                % Computing distance to the hyperplane for training data
                distance = (double(ZScoredData(train,:)) * w + bias) / norm(w);
                % Storing distances
                Distance(i,train) = distance;

                % Accuracy
                Accuracy(i) = sum(computedLabels == Labels(test)) / length(computedLabels) * 100;
             end

             Acc(MaskNum, SubNum,Permu)=mean(Accuracy) ;
             clear Beta_Pl Beta_Pl_r Beta_Pl_M NanVoxel_Pl Data_Pl DataF_Pl Pl
             clear Beta_Nt Beta_Nt_r Beta_Nt_M NanVoxel_Nt Data_Nt DataF_Nt Nt
             clear Beta_Up Beta_Up_r Beta_Up_M NanVoxel_Up Data_Up DataF_Up Up
             clear CombinedData ZScoredData weight 

        end
    end
end

% Average across the permutation cases (3rd dimension)
avgData = mean(Acc, 3); % This will give you a 2x20 array

% Plotting the boxplots
figure;
boxplot(avgData', 'Labels', {'ROI 1', 'ROI 2'});
hold on;

% Calculate and plot the mean for each ROI
means = mean(avgData, 2);
plot([1, 2], means, 'r*', 'MarkerSize', 10);

% Add labels and title
xlabel('ROI');
ylabel('Average Value');
title('Boxplot with Mean for Each ROI');

% Show the plot
hold off;

%  disp(['Accuracy: ', num2str(mean(Acc(:)))]);
%  
% figure
% A=[mean(NCC,1)*100;mean(PCC,1)*100;mean(Acc,1)];
% bar(A')
% 
% legend('Up','Nt','meanAcc')
% bar(mean(NCC,1)*100,'r')
% hold on
% bar(mean(PCC,1)*100,'b')
% bar(mean(Acc,1),'g')
% legend('Up','Nt','meanAcc')
% Acc2=mean(Acc,3);
% Dis2=mean(Dis,4);
% 
% 
% [FDR, Q] = mafdr(PValues);





