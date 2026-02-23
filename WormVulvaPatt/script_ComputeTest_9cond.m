load('VulvalDevelopmentPropsClass.mat')


numparamTrain=1000;

conditions=1:9;
NumCond=numel(conditions);

NumCondTotal=numel(unique(MatrixPropElena(:,end)));
NumParam=size(MatrixPropElena,1)/NumCondTotal;

if isfile('ML_Analysis_Prop_9_All_Test_1000.mat')
    load('ML_Analysis_Prop_9_All_Test_1000.mat')
    it0=it;
else
    ResultsPropTrain = zeros(1000, 1);
    ResultsPropTest = zeros(1000, 1);
    save('ML_Analysis_Prop_9_All_Test_1000.mat', 'ResultsPropTrain', 'ResultsPropTest', 'conditions');
    it0=0;
end

for it=it0+1:20
    ResultsPropTrainTemp = zeros(50, 1);
    ResultsPropTestTemp = zeros(50, 1);
    parfor rep=1:50

        %%Generate training
        iparam = randperm(NumParam, numparamTrain);
        PropsTrain=zeros(numparamTrain*NumCond, size(MatrixPropElena,2));
        for i=1:numparamTrain
            for Condidx=1:NumCond
                aux=MatrixPropElena(MatrixPropElena(:,end)==conditions(Condidx),:);
                PropsTrain((i-1)*NumCond+Condidx,:)=aux(iparam(i),:);
            end
        end
        PropsTrainRed=PropsTrain(:,var(PropsTrain)>0.01);

        iparamT = randperm(NumParam-numparamTrain, numparamTrain);
        IndicesTest=setdiff(1:NumParam, iparam);
        PropsTest=zeros(size(PropsTrain));
        for i=1:numparamTrain
            for Condidx=1:NumCond
                aux=MatrixPropElena(MatrixPropElena(:,end)==conditions(Condidx),:);
                PropsTest((i-1)*NumCond+Condidx,:)=aux(IndicesTest(iparamT(i)),:);
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
    save('ML_Analysis_Prop_9_All_Test_1000.mat', 'ResultsPropTrain', 'ResultsPropTest', 'conditions', 'it');
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
