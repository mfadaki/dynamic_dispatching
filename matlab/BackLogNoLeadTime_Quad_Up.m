DSampleUpSizeLarge=1000;
DSampleLargeLowUp=GenDemand(D, DSampleUpSizeLarge);
ActionSample=lbA+(ubA-lbA)*(0:0.01:1)';

num_stage=300;
num_exp=100;
%uniform initial state distribution
cost=[];
for i=1:num_exp
    s=rand.*(ubS-lbS)+lbS;
    cumcost=0;
    for t=1:num_stage
        BackLogNoLeadTime_Quad_compPolicy;
        tempdemand=GenDemand(D, 1);
        tempmatrix=s+qR-tempdemand;
        newS=min(max(tempmatrix,lbS),ubS);
        tempcost=qR*cR+max(newS,0)*hcost+max(-newS,0)*bcost;
        tempcost=tempcost+max(tempmatrix-ubS,0)*dcost+max(lbS-tempmatrix,0)*scost;
        cumcost=cumcost+gamma^(t-1)*tempcost;
        oldS=s;
        s=newS;
        %[i oldS qR tempcost cumcost]
        if (gamma^(t-1)*tempcost)/cumcost<0.00001
            break
        end
    end
    cost=[cost; cumcost];
    [i cumcost mean(cost) std(cost)/sqrt(i)]
    if (std(cost)/sqrt(i)<mean(cost)*0.001) & (length(cost)>1)
        break
    end
end
UB_temp=mean(cost);