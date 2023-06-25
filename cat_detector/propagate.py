def propagate(w, b, X, Y):
    """
    Implement the cost function and its gradient for the propagation explained above

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

    Return:
    grads -- dictionary containing the gradients of the weights and bias
            (dw -- gradient of the loss with respect to w, thus same shape as w)
            (db -- gradient of the loss with respect to b, thus same shape as b)
    cost -- negative log-likelihood cost for logistic regression
    
    Tips:
    - Write your code step by step for the propagation. np.log(), np.dot()
    """
  
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
    #print(type(grads["dw"]))
   

    # BACKWARD PROPAGATION (TO FIND GRAD)
    #(≈ 2 lines of code)
    


    
        
    
    return grads, cost  