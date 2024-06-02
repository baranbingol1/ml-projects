function [ thetas ] = logreg(X, y, max_iter)
% This function uses newton-rhapson method and hessian matrix to estimate thetas.
% First input X is the design matrix. Second input y is labels.
% Third input to the function is the max. iteration number.
    theta = zeros(size(X, 2), 1); % random init for theta vector   
    for iter=1:max_iter
   
        p = 1 ./ (1 + exp(-X * theta)); % sigmoid function (aka logistic)
        
        g = X' * (y - p); % gradient
    
        S = diag(p .* (1 - p)); 
        H = -X' * S * X; % hessian matrix
    
        thetas = theta - H \ g; % update thetas via newton-rhapson method
    end % end for
end % end function
