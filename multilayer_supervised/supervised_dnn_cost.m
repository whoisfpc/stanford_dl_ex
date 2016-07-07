function [ cost, grad, pred_prob] = supervised_dnn_cost( theta, ei, data, labels, pred_only)
%SPNETCOSTSLAVE Slave cost function for simple phone net
%   Does all the work of cost / gradient computation
%   Returns cost broken into cross-entropy, weight norm, and prox reg
%        components (ceCost, wCost, pCost)

%% default values
po = false;
if exist('pred_only','var')
  po = pred_only;
end;

%% reshape into network
stack = params2stack(theta, ei);
numHidden = numel(ei.layer_sizes) - 1;
hAct = cell(numHidden+1, 1);
gradStack = cell(numHidden+1, 1);
m = size(data, 2);
num_classes = ei.output_dim;
%% forward prop
%%% YOUR CODE HERE %%%
if isequal(ei.activation_fun, 'logistic')
    func = @sigmoid;
end
for l=1:numel(ei.layer_sizes)
    if l > 1
        if l == numHidden+1
            z = exp(stack{l}.W * hAct{l-1} + repmat(stack{l}.b, 1, m));
            hAct{l} = bsxfun(@rdivide, z, sum(z));
        else
            hAct{l} = func(stack{l}.W * hAct{l-1} + repmat(stack{l}.b, 1, m));
        end
    else
        hAct{l} = func(stack{l}.W * data + repmat(stack{l}.b, 1, m));
    end
end
%z = exp(hAct{end});
%pred_prob = bsxfun(@rdivide, z, sum(z));
pred_prob = hAct{end};
%% return here if only predictions desired.
if po
  cost = -1; ceCost = -1; wCost = -1; numCorrect = -1;
  grad = [];  
  return;
end;

%% compute cost
%%% YOUR CODE HERE %%%
y = full(sparse(labels, 1:m, 1, num_classes, m));
t = y .* pred_prob;
cost = -sum(log(t(t~=0)));
%% compute gradients using backpropagation
%%% YOUR CODE HERE %%%
delta = cell(numHidden+1, 1);
g_w = cell(size(delta));
g_b = cell(size(delta));
for l=numHidden+1:-1:1
    if l < numHidden+1
        delta{l} = (stack{l+1}.W' * delta{l+1}) .* (hAct{l} .* (1-hAct{l}));
    else
        delta{l} = -(y - pred_prob);
    end
    if l > 1
        g_w{l} = delta{l} * hAct{l-1}';
    else
        g_w{l} = delta{l} * data';
    end
    g_b{l} = sum(delta{l}, 2);
end
%% compute weight penalty cost and gradient for non-bias terms
%%% YOUR CODE HERE %%%
for l=1:length(gradStack)
    gradStack{l}.W = 1/m * g_w{l} + ei.lambda * stack{l}.W;
    gradStack{l}.b = 1/m * g_b{l};
end

%% reshape gradients into vector
[grad] = stack2params(gradStack);
end

function h=sigmoid(a)
  h=1./(1+exp(-a));
end


