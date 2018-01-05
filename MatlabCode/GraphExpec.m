function [mi,mx]=GraphExpec(y,yEst,k,str,w)
% regression graph that plots 2 sets of variables {y,yEst} against each
% other & automatically scales axes
subplot(2,4,k),  hold off cla, plot(y,yEst,'k.','LineWidth',2,'MarkerSize',15); 
hold on, plot(y,y,'Color',[0.7 0.7 0.7],'LineWidth',2);
a=min([y' yEst']); mi=(1-(1.1-sign(a)))*abs(a);
mx=1.1*max([y' yEst']);
axis([mi mx mi mx]);
set(gca,'FontSize',18); xlabel(['sim E[' str ']']); ylabel(['ana E[' str ']']);
text(mi+w(1),mx+w(2),['r\approx' num2str(round(1000*corr(y,yEst))/1000)],'FontSize',18);

%%
% (c) 2016 Daniel Durstewitz, Dept. Theoretical Neuroscience, Central
% Institute of Mental Health, Heidelberg University
