close all
clc
clear all
load('output.txt')
find(output(:,8)==1)
output(find(output(:,8)==1),1:(end-1))
paper=output(find(output(:,8)==1),1:(end-1))
rock=output(find(output(:,8)==2),1:(end-1))
scissors=output(find(output(:,8)==3),1:(end-1))
figure
scatter(rock(:,1),rock(:,2),'bx','linewidth',3)
hold on
scatter(paper(:,1),paper(:,2),'rx','linewidth',3)
hold on
scatter(scissors(:,1),scissors(:,2),'gx','linewidth',3)
hold off

[mu,E,lambda,p]=getEigenvectors(output(:,2:end));

p=[0;p];
plot(p','LineWidth', 2,'DisplayName', ...
    'Cumulative Percentage of Eigenvalues')
set(gca,'XTick',[0:10:60, 69])
set(gca,'YLim',[0 1.1])
xlabel('Component number')
ylabel('Cumulative Percentage')
figure
components=E(:,1:1);
data = (paper-repmat(mu',size(paper,1),1))*components;
plot(data(:,1),'r','linewidth',3)
hold on
data = (rock-repmat(mu',size(rock,1),1))*components;
plot(data(:,1),'b','linewidth',3)
hold on
data = (scissors-repmat(mu',size(scissors,1),1))*components;
plot(data(:,1),'g','linewidth',3)