addpath('C:\Program Files\IBM\ILOG\CPLEX_Studio1210\cplex\matlab\x64_win64');
addpath('C:\Program Files\IBM\ILOG\CPLEX_Studio1210\cplex\examples\src\matlab');
warning off MATLAB:lang:badlyScopedReturnValue

BackLogNoLeadTime_Quad_Setup;
InitialSampeSize=20000;

tic;
InitialSample=[diag(ubS-lbS)*rand(1,InitialSampeSize)+kron(lbS,ones(1,InitialSampeSize));
               diag(ubA-lbA)*rand(1,InitialSampeSize)+lbA;];
          
flp=[bphi0;bphi1;reshape(bphi2,[],1)];
blp=zeros(InitialSampeSize,1);


DSampleSizeLarge=80000;
DSampleLarge=GenDemand(D, DSampleSizeLarge);
LPobj=[];
Time=[];
ThetaTime=[];
for i=1:InitialSampeSize
    i
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
    if mod(i,500)==0
        [tempx,fval,exitflag,output] = cplexlp(-flp,Alp(1:i,:),blp(1:i),[],[],lbtheta,ubtheta);
        theta=tempx;
        thetabar=theta;
        LPobj=[LPobj;-fval];
        temptime=toc;
        Time=[Time;temptime];
        [i,temptime,-fval]
        ThetaTime=[ThetaTime;thetabar'];
    end
end
[tempx,fval,exitflag,output] = cplexlp(-flp,Alp,blp,[],[],lbtheta,ubtheta);
-fval
%output
theta=tempx;
thetabar=tempx;
save('BackLogNoLeadTimeLPInitialSampeSize20000D80000.mat','Alp','blp','DSampleSizeLarge','InitialSampeSize',...
    'InitialSample','DSampleLarge','ThetaTime','Time','LPobj','thetabar');