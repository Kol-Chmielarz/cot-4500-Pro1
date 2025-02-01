import numpy as np
def approx_alg(x):

    x_rnd = np.ceil(x)

    x = x_rnd

    tol = 0.000001

    iter = 0

    diff = x  # Initial diff to enter the loop

    print(f"{iter}: {x}")

    while diff >= tol:
        iter += 1
        y = x
        x = (y / 2) + (1 / y)
        print(f"{iter}: {x}")
        diff = abs(x - y)

    print(f"\nConvergence after {iter} iterations")
    return x



def bisection_method(g, p0, tol, max_iter):
    i = 1  # Iteration counter
    results = []  # Store iteration results

    while i <= max_iter:
        p = g(p0)  # Compute next approximation
        results.append((i, p))

        if abs(p - p0) < tol:
            print(f"SUCCESS: Converged to {p} in {i} iterations.")
            return p, i, True, results  # Return the root, iterations, and success flag
        
        i += 1
        p0 = p  # Update p0 for the next iteration

    print("Diverged: Maximum iterations reached without convergence.")
    return p0, max_iter, False, results  # Return last computed value and failure flag


def fixed_point(g, p0, tol, max_iter):
    i = 1  # Iteration counter
    results = []  # Store all iterations for printing

    while i <= max_iter:
        p = g(p0)  # Compute new approximation

        # Print iteration result
        print(f"{i} : {p:.15f}")

        # Stop if NaN is encountered (Divergence Detected)
        if np.isnan(p) or np.isinf(p):
            print("\nResult diverges")
            print(f"Failure after {i} iterations\n")
            return p0, i, "DIVERGED", results

        results.append((i, p))  # Store iteration step

        # Check convergence criteria
        if abs(p - p0) < tol:
            print("\nSuccess after", i, "iterations\n")
            return p, i, "SUCCESS", results  # Ensure four values are returned

        i += 1
        p0 = p  # Update p0 for next iteration

    # If max_iter reached, return failure message
    print("\nResult diverges")
    print(f"Failure after {max_iter} iterations\n")
    return p0, max_iter, "DIVERGED", results




def newtons_method(f, df, p0, tol, max_iter):
    i = 0  # Iteration counter
    results = [(i, p0)]  # Store first guess

    while i < max_iter:
        f_p0 = f(p0)
        df_p0 = df(p0)

        # Check if derivative is zero to prevent division error
        if df_p0 == 0:
            print("\nUnsuccessful: Derivative is zero")
            return None, i, "FAILURE", results

        # Compute next approximation
        p_next = p0 - f_p0 / df_p0
        results.append((i+1, p_next))  # Store iteration step
        #print(f"{i+1} : {p_next:.10f}")

        # Check for convergence
        if abs(p_next - p0) < tol:
            print(f"\nSuccess: Root found at {p_next:.10f} after {i+1} iterations\n")
            return p_next, i+1, "SUCCESS", results

        i += 1
        p0 = p_next  # Update for next iteration

    print("\nUnsuccessful: Maximum iterations reached")
    return p0, max_iter, "FAILURE", results

