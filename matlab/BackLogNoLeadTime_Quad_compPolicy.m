tempmatrix=kron(s+ActionSample,ones(1,DSampleUpSizeLarge))-kron(ones(length(ActionSample),1),DSampleLargeLowUp');
newS=min(max(tempmatrix,lbS),ubS);
tempcost=ActionSample*(ones(1,DSampleUpSizeLarge)*cR)+max(newS,0)*hcost+max(-newS,0)*bcost;
tempcost=tempcost+max(tempmatrix-ubS,0)*dcost+max(lbS-tempmatrix,0)*scost;
tempcost=mean(tempcost,2);
tempcost=tempcost+gamma*(thetabar(2)*mean(newS,2)+thetabar(3)*mean(newS.^2/2,2));
[temp, tempid]=min(tempcost);
qR=ActionSample(tempid);
