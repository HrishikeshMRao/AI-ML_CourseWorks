def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    """
    This function optimizes w and b by running a gradient descent algorithm
    
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- True to print the loss every 100 steps
    
    Returns:
    params -- dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.
    
    Tips:
    You basically need to write down two steps and iterate through them:
        1) Calculate the cost and the gradient for the current parameters. Use propagate().
        2) Update the parameters using gradient descent rule for w and b.
    """
    
    w = copy.deepcopy(w)
    b = copy.deepcopy(b)
    
    costs = []
    
    for i in range(num_iterations):
        # (≈ 1 lines of code)
        # Cost and gradient calculation 
        m = X.shape[1]
    
        # FORWARD PROPAGATION (FROM X TO COST)
        #(≈ 2 lines of code)
        # compute activation
        A = 1/(1+np.exp(-(np.dot(w.T,X)+b)))
        # compute cost by using np.dot to perform multiplication. 
        # And don't use loops for the sum.
        cost = -((np.sum((Y*np.log(A)+(1-Y)*np.log(1-A)),axis = 1, keepdims = True)))/m                                
        # YOUR CODE STARTS HERE
        cost = np.squeeze(np.array(cost))
        #print(type(cost))
        dw = (np.dot(X,(A-Y).T))/m
        db = (np.float64(np.sum((A-Y),axis =1, keepdims =True)))/m
        grads = {"dw": dw ,
                 "db": db}
        # YOUR CODE STARTS HERE
        
        
        # YOUR CODE ENDS HERE
        
        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]
        
        # update rule (≈ 2 lines of code)
        w = w - learning_rate*dw
        b = b - learning_rate*db
        # YOUR CODE STARTS HERE
        
        
        # YOUR CODE ENDS HERE
        
        # Record the costs
        if i % 100 == 0:
            costs.append(cost)
        
            # Print the cost every 100 training iterations
            if print_cost:
                print ("Cost after iteration %i: %f" %(i, cost))
    
    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params, grads, costs