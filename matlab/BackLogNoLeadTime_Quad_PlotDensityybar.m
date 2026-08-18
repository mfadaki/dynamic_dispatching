load('BackLogNoLeadTime_FOMgamma95.mat')
iter_to_plot=[0, 10, 20, 50, 100, 200, 300, 500, 800, 1000];
averageleny=1000;
countplot=1;
for iter=iter_to_plot
%     if iter==0
%         t=iter+1;
%     else
%         t=iter;
%     end
    t=iter;
    if t<averageleny-1
        temprange=1:(t+1)*USampleSize;
    else
        temprange=((t+2-averageleny)*USampleSize+1):(t+1)*USampleSize;
    end 
    tempid=datasample(temprange,USampleSize);
    sampleSA=SAHist(tempid,:);
    subplot(2,5,countplot)
    hold on
    scatter(sampleSA(:,1), sampleSA(:,2), 10, "filled");
    xlim([-10 10])
    ylim([0 10])
    xlabel('State');
    ylabel('Action');
    title(strcat('Iteration',{' '},num2str(iter)));
    set(gca,'fontsize', 14)
    hold off
    countplot=countplot+1;
end
