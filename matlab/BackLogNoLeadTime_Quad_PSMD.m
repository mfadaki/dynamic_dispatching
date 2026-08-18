clear;
addpath('C:\Program Files\IBM\ILOG\CPLEX_Studio1210\cplex\matlab\x64_win64');
addpath('C:\Program Files\IBM\ILOG\CPLEX_Studio1210\cplex\examples\src\matlab');
warning off MATLAB:lang:badlyScopedReturnValue
options = cplexoptimset('cplex');
options.display='off';
load('BackLogNoLeadTimeLPInitialSampeSize20000D80000.mat', 'Alp');
load('BackLogNoLeadTimeLPInitialSampeSize20000D80000.mat', 'blp');
AlpLarge=Alp; %only for checking correctness, no influence on PSMD algorithm
blpLarge=blp; %only for checking correctness, no influence on PSMD algorithm
objexact=[];

BackLogNoLeadTime_Quad_Setup;
starttime=cputime;
tic

num_inner_iter=1000;
%num_iter_computeLU=100;
num_iter_computeLU=0; %Set to 0 if do not need to compute LBUB during PSMD.
averagelentheta=num_inner_iter;
averageleny=num_inner_iter;
USampleSizeLarge=400; %Sample size in MH  %100
USampleSize=200; %How many MH sample to use after throw away burning period. %50
DSampleSizeLarge=1000;
DSampleSize=50;
UsePreSampleD=1;
%UsePreSampleDEval=1;
ReSampleDforMH=0;
ReSampleSAforMH=0;

DSampleLarge=GenDemand(D, 1000);  %20000 %1000
if (1)
InitialSampeSize=100; %initial SA sample size %2000  %200
InitialSample=[diag(ubS-lbS)*rand(1,InitialSampeSize)+kron(lbS,ones(1,InitialSampeSize));
               diag(ubA-lbA)*rand(1,InitialSampeSize)+lbA;
                ];
flp=[bphi0;bphi1;reshape(bphi2,[],1)];
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
[tempx,fval,exitflag,output] = cplexlp(-flp,Alp,blp,[],[],lbtheta,ubtheta);
end

% meanAlp=mean(abs(Alp),1)';
% scaledAlp=Alp*diag(1./meanAlp);
% scaledbphi=bphi./meanAlp;
% scaledlbtheta=lbtheta.*meanAlp;
% scaledubtheta=ubtheta.*meanAlp;

theta=tempx;
%theta=zeros(3,1);
thetabar=theta;
Theta=zeros(num_inner_iter,3);
Theta(1,:)=theta;
Thetabar=zeros(num_inner_iter,3);
Thetabar(1,:)=thetabar;
innerobj=zeros(num_inner_iter,3);
weightcost=zeros(num_inner_iter,1);
weighttheta=zeros(num_inner_iter,3);
DSampleHist=zeros(num_inner_iter,DSampleSize);
DSampleHist=reshape(GenDemand(D, num_inner_iter*DSampleSize),num_inner_iter,DSampleSize);
totalcount=0;
SAHist=zeros(num_inner_iter*USampleSize,2);
temprecord=-Alp*theta+blp;
[tempmax tempid]=min(temprecord);
sampleSA=kron(ones(USampleSize,1),InitialSample(:,tempid)');
sampleSA=[diag(ubS-lbS)*rand(1,USampleSize)+kron(lbS,ones(1,USampleSize));
          diag(ubA-lbA)*rand(1,USampleSize)+lbA;
          ]';
SAHist(1:USampleSize,:)=sampleSA;
ubound=0;
lbound=0;
UB=[];
LB=[];
Time=[];
ThetaTime=[];
TotalCount=0;
etabarx=0.01;
etabary=0.01;
lambdabar=1; %lambdabar=1;
lambdabart=0.1*lambdabar;
if num_iter_computeLU~=0
    BackLogNoLeadTime_Quad_Low;
    BackLogNoLeadTime_Quad_Up;
    LB=[LB;LB_temp];   
    UB=[UB;UB_temp];
end



for t=0:num_inner_iter-1
    etax=1/sqrt(t+1)*etabarx;
    etay=1/sqrt(t+1)*etabary;
    lambda=1/sqrt(t+1)*lambdabar;
    lambdabart=0.1*lambdabar/sqrt(t+2);
    if UsePreSampleD==0
        DSampleHist(t+1,:)=GenDemand(D, DSampleSize);
    end
    weightcost=weightcost*1/(1+lambda*etay);
    weightcost(t+1,1)=-etay/(1+lambda*etay);
    weighttheta=weighttheta*1/(1+lambda*etay);
    weighttheta(t+1,:)=-etay/(1+lambda*etay)*theta;

    BackLogNoLeadTime_Quad_SampleSA;

    tempDemand=GenDemand(D, DSampleSizeLarge);
    tempmatrix=kron(sum(sampleSA,2),ones(1,DSampleSizeLarge))-kron(ones(USampleSize,1),tempDemand');
    newS=min(max(tempmatrix,lbS),ubS);
    tempcost=sampleSA(:,2)*(ones(1,DSampleSizeLarge)*cR)+max(newS,0)*hcost+max(-newS,0)*bcost;
    tempcost=tempcost+max(tempmatrix-ubS,0)*dcost+max(lbS-tempmatrix,0)*scost;
    tempcost=mean(mean(tempcost,2));
    tempcost=tempcost/(1-gamma);
    grad0=(gamma-1)/(1-gamma);
    grad1=mean(gamma*mean(newS,2)-sampleSA(:,1))/(1-gamma);
    grad2=mean(gamma*mean(newS.^2/2,2)-(sampleSA(:,1)).^2/2)/(1-gamma);
    grad=[grad0;grad1;grad2]+bphi;
    f=-etax*grad-theta;
    H=eye(3);
    [theta,fval,exitflag,output]  = cplexqp(H,f,[],[],[],[],lbtheta,ubtheta,[],options);
    if t<averagelentheta-1
        thetabar=(thetabar*(t+1)+theta)/(t+2);
    else
        thetabar=(thetabar*(averagelentheta)+theta-Theta(t+2-averagelentheta,:)')/(averagelentheta);
    end
    
    Theta(t+2,:)=theta';
    Thetabar(t+2,:)=thetabar';
    sampleSA=sampleSAnew;
    SAHist((t+1)*USampleSize+1:(t+2)*USampleSize,:)=sampleSA;   
    
    if t<averageleny-1
        temprange=1:(t+2)*USampleSize;
    else
        temprange=((t+3-averageleny)*USampleSize+1):(t+2)*USampleSize;
    end 
    
    temp=bphi(2:3)'*thetabar(2:3);
    temp=min(blpLarge-AlpLarge(:,2:3)*thetabar(2:3))/(1-gamma)+temp;
    objexact=[objexact;temp];
    %[t temp thetabar']
    
    if num_iter_computeLU~=0
        if mod(t+1,num_iter_computeLU)==0
            temptime=toc;
            Time=[Time;temptime];
            ThetaTime=[ThetaTime;thetabar];
            BackLogNoLeadTime_Quad_Low;
            BackLogNoLeadTime_Quad_Up;
            LB=[LB;LB_temp];   
            UB=[UB;UB_temp];
        end
        [t temp max(objexact) LB_temp max(LB) UB_temp min(UB)]
    else
        [t temp max(objexact)]
    end
end
endtime=cputime;
totaltime=endtime-starttime;
toc
if num_iter_computeLU~=0
    UBmin=UB;
    LBmax=LB;
    for i=1:length(LB) 
        LBmax(i)=max(LB(1:i));
    end
    for i=1:length(LB) 
        UBmin(i)=min(UB(1:i));
    end
end
save('BackLogNoLeadTime_FOMgamma95.mat')
