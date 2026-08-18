cR=20;%ordering cost or cR=10
hcost=2;%holding cost
bcost=10;%backlog cost
dcost=10;%disposal cost or dcost=100
scost=100;%salvage cost
D.Dist='TrunNorm';
D.para1=5;%mu
D.para2=2;%sigma
D.para3=10;%max demand

mulD=1;
lbS=[-D.para3*mulD;
    ];
ubS=[D.para3*mulD;
    ];
lbA=[0];
ubA=[D.para3*mulD;];

%lbtheta=[-Inf;-20;-20];
%ubtheta=[Inf;20;20];
lbtheta=[-Inf;-Inf;-Inf];
ubtheta=[Inf;Inf;Inf];

gamma=0.95;

%uniform initial state distribution
bphi0=1;
bphi1=(lbS+ubS)/2;
bphi2=bphi1*bphi1'/2+diag((ubS-lbS).^2)/24;
bphi=[bphi0;bphi1;bphi2];
