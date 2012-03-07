close all
clc
clear all
load('output.txt')
find(output(:,8)==1)
output(find(output(:,8)==1),1:(end-1))

output(:,(1:7)) =output(:,(1:7))- repmat(mean(output(:,1:7)),size(output(:,1:7), 1),1);
output(:,(1:7)) =output(:,(1:7))./repmat(std(output(:,(1:7))),size(output(:,(1:7)), 1),1);


paper=output(find(output(:,8)==1),1:(end-1));
rock=output(find(output(:,8)==2),1:(end-1));
scissors=output(find(output(:,8)==3),1:(end-1));
figure
scatter(rock(:,1),rock(:,2),'bx','linewidth',3);
hold on
scatter(paper(:,1),paper(:,2),'rx','linewidth',3);
hold on
scatter(scissors(:,1),scissors(:,2),'gx','linewidth',3);
hold off

[mu,E,lambda,p]=getEigenvectors(output(:,1:(end-1)));

p=[0;p];
plot(p','LineWidth', 2,'DisplayName', ...
    'Cumulative Percentage of Eigenvalues')
set(gca,'XTick',[0:10:60, 69])
set(gca,'YLim',[0 1.1])
xlabel('Component number')
ylabel('Cumulative Percentage')
figure
components=E(:,1:5);

data = (paper-repmat(mu',size(paper,1),1))*components;
plot3(data(:,1),data(:,2),data(:,3),'r','linewidth',3)
X=data;
y=1*ones(size(paper,1),1);
hold on

data = (rock-repmat(mu',size(rock,1),1))*components;
plot3(data(:,1),data(:,2),data(:,3),'b','linewidth',3)
X=[X;data];
y=[y;2*ones(size(rock,1),1)];
hold on

data = (scissors-repmat(mu',size(scissors,1),1))*components;
plot3(data(:,1),data(:,2),data(:,3),'g','linewidth',3)
X=[X; data];
y=[y;3*ones(size(scissors,1),1)];
size(X);
size(y);

% 
%X_T= [X(:,1:3),X(:,1).*X(:,2),X(:,2).*X(:,3),X(:,1).*X(:,3)];
X_T= [X(:,1:4),X(:,1).*X(:,2),X(:,2).*X(:,3),X(:,1).*X(:,3),X(:,4).*X(:,1),X(:,4).*X(:,2), X(:,4).*X(:,3)];
X_T=X(:,1:5)
% X_T=[X_T, y]
X=X_T;
rand_indices = randperm(size(X,1));
X_orig=X;
y_orig=y;

%
corrects=0;
falses=0;
preds=[];
y_tests=[];
for i=1:(size(X,1))
    
training_size=size(X_orig,1);
training=[rand_indices(1:i-1),rand_indices(i+1:end)];
X_train = X_orig(training, :);
y_train = y_orig(training, :);
% X=sel(:,1:(end-1));
% y=sel(:,end);
X_test=X_orig(rand_indices(i), :);
y_test=y_orig(rand_indices(i), :);

size(X_train)
size(y_train)

size(X_test)
size(y_test)
% %% Setup the parameters you will use for this part of the exercise
% %input_layer_size  = 9;  % 3 PCA
num_labels = 3;          % rock, papaer, scissors
%                           % (note that we have mapped "0" to label 10)
% 
% %% =========== Part 1: Loading and Visualizing Data =============
% %  We start the exercise by first loading and visualizing the dataset. 
% %  You will be working with a dataset that contains handwritten digits.
% %
% 
% % Load Training Data
% fprintf('Loading and Visualizing Data ...\n')
% 
X=X_train;
y=y_train;
m = size(X, 1);
% 
% 
% 
% fprintf('Program paused. Press enter to continue.\n');
% pause;
% 
% %% ============ Part 2: Vectorize Logistic Regression ============
% %  In this part of the exercise, you will reuse your logistic regression
% %  code from the last exercise. You task here is to make sure that your
% %  regularized logistic regression implementation is vectorized. After
% %  that, you will implement one-vs-all classification for the handwritten
% %  digit dataset.
% %
% 
% fprintf('\nTraining One-vs-All Logistic Regression...\n')
% 
lambda = 0.5;
[all_theta] = oneVsAll(X_train, y_train, num_labels, lambda);
% 
% fprintf('Program paused. Press enter to continue.\n');
% pause;
% 
% 
% %% ================ Part 3: Predict for One-Vs-All ================
% %  After ...
pred = predictOneVsAll(all_theta, X_test);
%
preds=[preds;pred];
y_tests=[y_tests;y_test];

if(pred==y_test)
    corrects=corrects+1;
else
    falses=falses+1;
end
all_theta

% 
    
end

mean(double(preds==y_tests)*100)
corrects/double(corrects+falses)