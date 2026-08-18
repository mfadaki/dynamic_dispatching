Cbar=log(1/20/10)-2*100*sqrt(200)+2*log(200)-log(1/3.1415);
USampleLowSizeLarge=400; %Sample size in MH
USampleLowSize=200; %How many MH sample to use after throw away burning period.
DSampleLowSizeLarge=1000;
DSampleLargeLow=GenDemand(D, DSampleLowSizeLarge);
InitialSampeLowSize=2000;
InitialSampeLow=[diag(ubS-lbS)*rand(1,InitialSampeLowSize)+kron(lbS,ones(1,InitialSampeLowSize));
                 diag(ubA-lbA)*rand(1,InitialSampeLowSize)+lbA;
                ];
Alp_Low=zeros(InitialSampeLowSize,1+1+1);
blp_Low=zeros(InitialSampeLowSize,1);
for i=1:InitialSampeLowSize
    unisampleS=InitialSampeLow(1,i);
    unisampleA=InitialSampeLow(2,i);
    snew=min(max(unisampleS+unisampleA-DSampleLargeLow,lbS),ubS);
    tempcost=unisampleA*cR+max(snew,0)*hcost+max(-snew,0)*bcost;
    tempcost=tempcost+max(unisampleS+unisampleA-DSampleLargeLow-ubS,0)*dcost+max(lbS+DSampleLargeLow-unisampleS-unisampleA,0)*scost;
    cost=mean(tempcost,2);
    Alp_Low(i,1)=1-gamma;
    Alp_Low(i,2)=InitialSampeLow(1,i)'-gamma*mean(snew);
    Alp_Low(i,3)=InitialSampeLow(1,i)*InitialSampeLow(1,i)'/2-gamma*mean(snew.*snew)/2;
    blp_Low(i)=mean(cost);
end

tempDemand=GenDemand(D, DSampleLowSizeLarge);
temprecord=(blp_Low-Alp_Low(:,2:3)*thetabar(2:3))/(1-gamma);
[tempmax tempid]=max(temprecord);
logpdf=@(x)BackLogNoLeadTime_Quad_Low_logpdf(x,tempDemand,thetabar,lbS,lbA,ubS,ubA,gamma,hcost,bcost,dcost,scost,cR,lambdabart);
proppdf =@(x,y) mvnpdf(x-y,zeros(2),eye(2)*0.2);
proprnd = @(x) x + randn(1,2)*0.2;
tempSA = mhsample(InitialSampeLow(:,tempid)',USampleLowSizeLarge,'logpdf',logpdf,'proprnd',proprnd,'symmetric',1);
sampleSALow =tempSA(USampleLowSizeLarge-USampleLowSize+1:USampleLowSizeLarge,:);

tempmatrix=kron(sum(sampleSALow,2),ones(1,DSampleLowSizeLarge))-kron(ones(USampleLowSize,1),tempDemand');
newS=min(max(tempmatrix,lbS),ubS);
tempcost=sampleSALow(:,2)*(ones(1,DSampleLowSizeLarge)*cR)+max(newS,0)*hcost+max(-newS,0)*bcost;
tempcost=tempcost+max(tempmatrix-ubS,0)*dcost+max(lbS-tempmatrix,0)*scost;
tempcost=mean(mean(tempcost,2));
tempcost=tempcost/(1-gamma);
grad0=(gamma-1)/(1-gamma);
grad1=mean(gamma*mean(newS,2)-sampleSALow(:,1))/(1-gamma);
grad2=mean(gamma*mean(newS.^2/2,2)-(sampleSALow(:,1)).^2/2)/(1-gamma);
fvalue=tempcost+thetabar(2)*grad1+thetabar(3)*grad2+thetabar(2)*bphi(2)+thetabar(3)*bphi(3);
LB_temp=fvalue+lambdabart*Cbar+2*lambdabart*log(lambdabart);