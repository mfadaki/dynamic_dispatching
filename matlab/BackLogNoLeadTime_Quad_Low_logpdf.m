function [temp]=BackLogNoLeadTime_Quad_Low_logpdf(x,tempDemand,thetabar,lbS,lbA,ubS,ubA,gamma,hcost,bcost,dcost,scost,cR,lambdabart)
unisampleS=x(1);
unisampleA=x(2);
if length(find(unisampleS>ubS'))>0
    temp=-Inf;
    return;
end
if length(find(unisampleS<lbS'))>0
    temp=-Inf;
    return;
end
if length(find(unisampleA>ubA))>0
    temp=-Inf;
    return;
end
if length(find(unisampleA<lbA))>0
    temp=-Inf;
    return;
end

tempmatrix=unisampleS+unisampleA-tempDemand;
newS=min(max(tempmatrix,lbS),ubS);
tempcost=max(newS,0)*hcost+max(-newS,0)*bcost;
tempcost=unisampleA*cR+mean(tempcost,1);
tempmatrix=max(tempmatrix-ubS,0)*dcost+max(lbS-tempmatrix,0)*scost;
cost=tempcost+mean(tempmatrix,1);
basefun0=(gamma-1);
basefun1=gamma*mean(newS)-unisampleS;
basefun2=gamma*mean(newS.^2/2)-unisampleS.^2/2;
temp=cost+basefun1*thetabar(2)+basefun2*thetabar(3);
temp=-temp/(1-gamma)/lambdabart;
%temp=temp/(1-gamma)-tempmax;
%temp=max(min(temp,100),-100);
%temp=exp(temp);
end