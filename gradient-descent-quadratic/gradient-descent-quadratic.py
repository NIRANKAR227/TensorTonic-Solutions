def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    """
    Return final x after 'steps' iterations.
    """
    # Write code here
    while steps>1:
        dfdx=2*x0*a+b
        x0=x0-lr*dfdx
        steps-=1
    return x0