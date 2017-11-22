function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%





X = [ones(m,1) X];

new_y = eye(num_labels);

y = new_y(y,:);

a1 = X;

z2 = a1 * Theta1';

a2 = sigmoid(z2);

a2 = [ones(m,1) a2];

z3 = a2 * Theta2';

a3 = sigmoid(z3);


htheta = a3;





%	for k=1:size(htheta,2)
%		yk = y == k %yk is 5000*1 matrix. Make yk 1 where y is equal to k(class).
%		hthetak = htheta(:,k) %htheta is 5000*10 matrix. Make new hthetak matrix with 5000 * 1 wrt k(class)
%		Jk = 1 / m * sum(-yk .* log(hthetak) - (1 - yk) .* log(1 - hthetak)); %Multiply y column with htheta column matrix. It will be 5000*1 matrix
%   	J = J + Jk; %Add all columns to get total error
%		end


%Without for loop
%Two sum functions as we have 2D matrix

J = 1/m .* sum(sum((-y) .* log(htheta) - (1-y) .* log(1-htheta)));

%Calculation of regularization reg term
%First remove 1st column because we do not perform regularization on theta0
%So we set first column as 0. 

temp1 = Theta1
temp1(:,1) = 0
temp2 = Theta2
temp2(:,1) = 0

%Theta1 and Theta2 are 2D matrix. We need squared sum of all values. 

reg_term = lambda /(2*m) * [sum(sum(temp1 .^ 2)) + sum(sum(temp2 .^ 2))]

%Add regularization term with previous J value to get regularized cost 
J = J + reg_term



% -------------------------------------------------------------

%PART 2
tri_delta_two = 0
tri_delta_one = 0
%for t=1:m
delta_three = a3 - y
z2 = [ones(m,1) z2]
delta_two = delta_three * Theta2 .* sigmoidGradient(z2)
delta_two = delta_two(:,2:end) %remove delta0 of second layer
tri_delta_one = tri_delta_one + delta_two' * a1
tri_delta_two = tri_delta_two + delta_three' * a2

Theta1_grad = 1/m .* tri_delta_one
Theta2_grad = 1/m .* tri_delta_two

Theta1_grad = 1/m .* tri_delta_one + (lambda/m) * temp1
Theta1_grad(1) = 1/m .* tri_delta_one(1)


Theta2_grad = 1/m .* tri_delta_two + (lambda/m) * temp2
Theta2_grad(1) = 1/m .* tri_delta_two(1)



% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
