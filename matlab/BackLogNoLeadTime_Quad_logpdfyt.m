function [temp]=BackLogNoLeadTime_Quad_logpdfyt(x,t,weightcost, weighttheta,DSampleHist,lbS,lbA,ubS,ubA,gamma,hcost,bcost,dcost,scost,cR,tempmax)
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

tempmatrix=unisampleS+unisampleA-DSampleHist(1:t+1,:);
newS=min(max(tempmatrix,lbS),ubS);
tempcost=max(newS,0)*hcost+max(-newS,0)*bcost;
tempcost=unisampleA*cR+mean(tempcost',1);
tempmatrix=max(tempmatrix-ubS,0)*dcost+max(lbS-tempmatrix,0)*scost;
cost=tempcost+mean(tempmatrix',1);
basefun0=(gamma-1);
basefun1=gamma*mean(newS,2)-unisampleS;
basefun2=gamma*mean(newS.^2/2,2)-unisampleS.^2/2;
temp=cost*weightcost(1:t+1)+basefun1'*weighttheta(1:t+1,2)+basefun2'*weighttheta(1:t+1,3);
temp=temp/(1-gamma);
%temp=temp/(1-gamma)-tempmax;
%temp=max(min(temp,100),-100);
%temp=exp(temp);
end