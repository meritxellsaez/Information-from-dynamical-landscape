load('ProportionsAll.mat')


times=1:6;
NumTimes = numel(times);
dimsTimes = zeros(1,2*NumTimes);
for i=1:NumTimes
    dimsTimes(2*i-1:2*i)=[2*times(i)-1, 2*times(i)];
end

numparamTrain=1000;

Cases={1:11};
conditions=Cases{1};
NumCond=numel(conditions);
if isfile('ML_Analysis_Prop_11_All_Test_1000.mat')
    load('ML_Analysis_Prop_11_All_Test_1000.mat')
    it0=it;
else
    ResultsPropTrain = zeros(1000, numel(Cases));
    ResultsPropTest = zeros(1000, numel(Cases));
    save('ML_Analysis_Prop_11_All_Test_1000.mat', 'ResultsPropTrain', 'ResultsPropTest', 'Cases');
    it0=0;
end
for it=it0+1:20
    ResultsPropTrainTemp = zeros(50, numel(Cases));
    ResultsPropTestTemp = zeros(50, numel(Cases));
    parfor rep=1:50

        %%Generate training
        iparam = randperm(10000, numparamTrain);

        PropsTrain=zeros(numparamTrain*NumCond, NumTimes*numComp+1);
        for i=1:numparamTrain
            for Condidx=1:NumCond

                PropsTrain((i-1)*NumCond+Condidx,1:end-1)=Props(Condidx, iparam(i),:);

                PropsTrain((i-1)*NumCond+Condidx,end)=Condidx;
            end
        end
        PropsTrainRed=PropsTrain(:,var(PropsTrain)>0.01);

        %%Generate test
        iparamT = randperm(10000-numparamTrain, numparamTrain);
        IndicesTest=setdiff(1:10000, iparam);
        PropsTest=zeros(numparamTrain*NumCond, NumTimes*numComp+1);
        for i=1:numparamTrain
            for Condidx=1:NumCond

                PropsTest((i-1)*NumCond+Condidx,1:end-1)=Props(Condidx, IndicesTest(iparamT(i)),:);

                PropsTest((i-1)*NumCond+Condidx,end)=Condidx;
            end
        end
        PropsTestRed=PropsTest(:,var(PropsTrain)>0.01);

        %%Train classifier
        if size(PropsTrainRed,2)>1
            MdlLinear = fitcecoc(PropsTrainRed(:,1:end-1),squeeze(PropsTrain(:,end)),'OptimizeHyperparameters', 'auto');

            %%Compute info

            [EachInfo]=Prediction(MdlLinear,PropsTrainRed);

            [EachInfoT, cnfm1, ~]=Prediction(MdlLinear,PropsTestRed);

            ResultsPropTrainTemp(rep)=EachInfo;
            ResultsPropTestTemp(rep)=EachInfoT;

        end

    end
    indxmatrix=50*(it-1)+(1:50);
    ResultsPropTrain(indxmatrix,:)=ResultsPropTrainTemp;
    ResultsPropTest(indxmatrix,:)=ResultsPropTestTemp;
    save('ML_Analysis_Prop_11_All_Test_1000.mat', 'ResultsPropTrain', 'ResultsPropTest', 'Cases', 'it');
end



function [infoML, cnfm1, labels]=Prediction(Mdl, Data)

    labels = predict(Mdl,Data(:,1:end-1));
    
    cnfm= confusionmat(Data(:,end),round(labels));
    cnfm1=cnfm/size(Data,1);
    
    sumRows = repmat(sum(cnfm1,2),1,size(cnfm,1));
    sumColumns = repmat(sum(cnfm1,1), size(cnfm,1),1);
    L=log2(cnfm1./(sumColumns.*sumRows));
    L(isinf(L))=0*sumColumns(isinf(L));
    L(isnan(L))=0*sumColumns(isnan(L));
    infoML=sum(cnfm1.*L,'all');

end
