function [mu, E, lambda, p ] = getEigenvectors(sequence)
%calculate mean of data
mu=mean(sequence)';
%eigenvalues and eigenvectors of covariance of dataset
[E,lambda,~]=svd(cov(sequence));
lambda=diag(lambda)
%Calculating cumulative percentage of eigenvalues.
p=cumsum(lambda)/sum(lambda);
end