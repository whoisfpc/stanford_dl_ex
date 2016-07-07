function [f,g] = softmax_regression(theta, X,y)
  %
  % Arguments:
  %   theta - A vector containing the parameter values to optimize.
  %       In minFunc, theta is reshaped to a long vector.  So we need to
  %       resize it to an n-by-(num_classes-1) matrix.
  %       Recall that we assume theta(:,num_classes) = 0.
  %
  %   X - The examples stored in a matrix.  
  %       X(i,j) is the i'th coordinate of the j'th example.
  %   y - The label for each example.  y(j) is the j'th example's label.
  %
  m=size(X,2);
  n=size(X,1);

  % theta is a vector;  need to reshape to n x num_classes.
  theta=reshape(theta, n, []);
  num_classes=size(theta,2)+1;
  
  % initialize objective value and gradient.
  f = 0;
  g = zeros(size(theta));

  %
  % TODO:  Compute the softmax objective function and gradient using vectorized code.
  %        Store the objective function value in 'f', and the gradient in 'g'.
  %        Before returning g, make sure you form it back into a vector with g=g(:);
  %
%%% YOUR CODE HERE %%%
  z = exp([theta, zeros(n,1)]' * X); % 10 X m matrix
  norm_term = sum(z); % 1 X m vector
  h = bsxfun(@rdivide, z, norm_term); % probability from 1 to 10 for m samples
  groundTruth = full(sparse(y, 1:m, 1, num_classes, m));
  t = groundTruth .* h;
  f = - sum(log(t(t~=0)));
  g = -X * (groundTruth - h)';
  g = g(:,1:end-1);
%   theta_g = gpuArray(theta);
%   X_g = gpuArray(X);
%   y_g = gpuArray(full(sparse(y, 1:m, 1)));
%   z = exp([theta_g, zeros(n,1,'gpuArray')]' * X_g);
%   norm_term = sum(z);
%   h = bsxfun(@rdivide, z, norm_term);
%   t = y_g .* h;
%   f_g = - sum(log(t(t~=0)));
%   g_g = -X_g * (y_g - h)';
%   g_g = g_g(:,1:end-1);
%   f = gather(f_g);
%   g = gather(g_g);
  g=g(:); % make gradient a vector for minFunc

