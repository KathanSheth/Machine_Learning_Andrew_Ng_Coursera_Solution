function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.


%for i = 1:m
		
%	h = theta(1) + theta(2) * X(i,2); %Here theta(1) is theta 0 in equation and theta(2) is theta1 in eqation. This line will equate h = theta0 + theta1*X1
%	J = J + (h - y(i))*(h-y(i));	% This will add Squared error to J function
%endfor
		
			

%J = 1/(2*m)*(J) %Return J. Devide by 2m
		
		%In above code I have used loop, which is not good!! Below is the vectorized solution

	h=X*theta;
	sqErrors=(h-y).^2;
	J=1/(2*m)*sum(sqErrors);


% =========================================================================

end
