"""
Utility functions for research.ipynb.

Author: Ofir Paz
Version: (23/10/2023)
"""

def sign(z, numpy_module):
    """
    Compute the sign(s) of a number(s).
    (could be an array of numbers)

    Parameters:
    z               --  Input number(s).
    numpy_module    --  The NumPy module.

    Returns:
    sign_s  --  Sign(s) of the input number(s) (+1 if positive, -1 if negative).
    """

    sign_s = numpy_module.where(z > 0, +1, -1)
    return sign_s

def evalEin(w, X_set, y_set, numpy_module) -> float:
    """
    Evaluates the In Sample Error of a given weight vector(s) on a given dataset.

    Parameters:
    w       --  The weight vector(s) to check its in sample error.
    X_set   --  The input vectors of the dataset (stacked coloumns).
    y_set   --  The labels corresponding to the input vectors of the set.
    numpy_module    --  The NumPy module.

    Returns:
    E_in    --  The In Sample Error of the weight vector on the dataset.
    """

    # Calculate predictions of input set X based on weight vector w
    y_predict_of_w = sign(numpy_module.dot(w.T, X_set), numpy_module)

    # Create an array that has 1 for a misclassification and 0 for correct classification
    misclassifies = (y_predict_of_w != y_set).astype(int)

    # Extract the number of training examples
    num_examples = X_set.shape[1]

    # Calculate E_in
    E_in = numpy_module.sum(misclassifies, dtype=float, axis=1) / num_examples

    return E_in