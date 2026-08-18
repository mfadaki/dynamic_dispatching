if length(DSampleLarge)~=1000
    ReSampleDforMH=1;
end
if InitialSampeSize~=200
    ReSampleSAforMH=1;
end
if ReSampleDforMH==1
    DSampleLarge=GenDemand(D, 1000);  %20000 %1000
    ReSampleDforMH=0;
end
if ReSampleSAforMH==1
    ReSampleDforMH=0;
    InitialSampeSize=200; %initial SA sample size %2000  %200
    InitialSample=[diag(ubS-lbS)*rand(1,InitialSampeSize)+kron(lbS,ones(1,InitialSampeSize));
                   diag(ubA-lbA)*rand(1,InitialSampeSize)+lbA;
                    ];
    Alp=zeros(InitialSampeSize,1+1+1);
    blp=zeros(InitialSampeSize,1);
    for i=1:InitialSampeSize
        i;
        unisampleS=InitialSample(1,i);
        unisampleA=InitialSample(2,i);
        snew=min(max(unisampleS+unisampleA-DSampleLarge,lbS),ubS);
        tempcost=unisampleA*cR+max(snew,0)*hcost+max(-snew,0)*bcost;
        tempcost=tempcost+max(unisampleS+unisampleA-DSampleLarge-ubS,0)*dcost+max(lbS+DSampleLarge-unisampleS-unisampleA,0)*scost;
        cost=mean(tempcost,2);
        Alp(i,1)=1-gamma;
        Alp(i,2)=InitialSample(1,i)'-gamma*mean(snew);
        Alp(i,3)=InitialSample(1,i)*InitialSample(1,i)'/2-gamma*mean(snew.*snew)/2;
        blp(i)=mean(cost);
    end
end

temprecord=(blp*sum(weightcost(1:t+1))-Alp(:,2:3)*sum(weighttheta(1:t+1,2:3),1)')/(1-gamma);
[tempmax tempid]=max(temprecord);
%pdf=@(x)BackLogNoLeadTime_Quad_pdfyt(x,t,weightcost, weighttheta,DSampleHist,lbS,lbA,ubS,ubA,gamma,hcost,bcost,dcost,scost,cR,tempmax);
logpdf=@(x)BackLogNoLeadTime_Quad_logpdfyt(x,t,weightcost, weighttheta,DSampleHist,lbS,lbA,ubS,ubA,gamma,hcost,bcost,dcost,scost,cR,tempmax);
proppdf =@(x,y) mvnpdf(x-y,zeros(2),eye(2)*0.2);
proprnd = @(x) x + randn(1,2)*0.2;
%     proppdf =@(x,y) mvnpdf(x-y,zeros(2),eye(2)*0.1);
%     proprnd = @(x) x + randn(1,2)*0.1;
%   tempSA = mhsample(InitialSample(:,tempid)',USampleSizeLarge,'pdf',pdf,'proprnd',proprnd,'symmetric',1);
tempSA = mhsample(InitialSample(:,tempid)',USampleSizeLarge,'logpdf',logpdf,'proprnd',proprnd,'symmetric',1);
sampleSAnew=tempSA(USampleSizeLarge-USampleSize+1:USampleSizeLarge,:);

%     temprecord=(blp*sum(weightcost(1:t+1))-Alp(:,2:3)*sum(weighttheta(1:t+1,2:3),1)')/(1-gamma);
%     [tempmax tempid]=sort(temprecord, 'descend');
%     tempSA = mhsample(InitialSample(:,tempid(1:5))',USampleSize/5,'logpdf',logpdf,'proprnd',proprnd,'symmetric',1, 'burnin',1,'thin',1,'nchain',5);
%     sampleSAnew = [tempSA(:,:,1);
%                    tempSA(:,:,2);
%                    tempSA(:,:,3);
%                    tempSA(:,:,4);
%                    tempSA(:,:,5)];
    