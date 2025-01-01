import numpy as np
from random import random

class MLP:
    def __init__(self, num_inputs=3, num_outputs=2, num_hidden=[3, 5]):
        """
        Initialize a Multi-Layer Perceptron (MLP) neural network.

        This constructor initializes an MLP with the specified number of input neurons, 
        output neurons, and hidden layers. The weights are randomly initialized for each 
        layer, and arrays for storing the activations and derivatives are also created.

        The MLP architecture is defined as:
        - `num_inputs`: The number of neurons in the input layer.
        - `num_hidden`: A list defining the number of neurons in each hidden layer.
        - `num_outputs`: The number of neurons in the output layer.

        Parameters:
        ----------
        num_inputs : int, optional, default=3
            The number of neurons in the input layer.

        num_outputs : int, optional, default=2
            The number of neurons in the output layer.

        num_hidden : list of int, optional, default=[3, 5]
            A list defining the number of neurons in each hidden layer. Can have any 
            number of elements to specify multiple hidden layers.

        Attributes:
        ----------
        weights : list of numpy.ndarray
            A list of weight matrices for each layer connection. Each matrix is a 
            2D numpy array representing the weights between two adjacent layers.

        derivatives : list of numpy.ndarray
            A list of arrays to store the gradients (derivatives) of the loss function 
            with respect to the weights for each layer.

        activations : list of numpy.ndarray
            A list of arrays to store the activations (outputs) for each layer. Each 
            layer's activation is a 1D numpy array of the size of that layer.

        """
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_hidden = num_hidden

        layers = [self.num_inputs] + self.num_hidden + [self.num_outputs]

        # create random connection weights for the layers
        weights = []
        for i in range(len(layers) - 1):
            w = np.random.rand(layers[i], layers[i + 1])
            weights.append(w)
        self.weights = weights

        # save derivatives per layer
        derivatives = []
        for i in range(len(layers) - 1):
            d = np.zeros((layers[i], layers[i + 1]))
            derivatives.append(d)
        self.derivatives = derivatives

        # save activations per layer
        activations = []
        for i in range(len(layers)):
            a = np.zeros(layers[i])
            activations.append(a)
        self.activations = activations

    def _sigmoid(self, x):
        """
        Compute the sigmoid of a given input.

        The sigmoid function is a mathematical function that maps any real-valued number 
        into the range (0, 1). It is often used in machine learning and neural networks 
        to introduce non-linearity in models.

        Parameters:
        ----------
        x : float or numpy.ndarray
            The input value or array of values to which the sigmoid function will be applied.

        Returns:
        -------
        float or numpy.ndarray
            The sigmoid of the input value(s). If the input is a single value, the output 
            will be a single float; if the input is an array, the output will be an array 
            of the same shape with the sigmoid applied element-wise.
        """
        return 1 / (1 + np.exp(-x))
    
    def _sigmoid_derivative(self, x):
        """
        Compute the derivative of the sigmoid function.

        The derivative of the sigmoid function is important in optimization algorithms, 
        particularly in backpropagation for neural networks. It helps to determine how 
        much change in the input will affect the output, allowing the model to learn and 
        adjust weights accordingly.

        The derivative of the sigmoid function is: f'(x) = x * (1 - x), where x is the 
        sigmoid output.

        Parameters:
        ----------
        x : float or numpy.ndarray
            The input value or array of sigmoid outputs for which the derivative is calculated.

        Returns:
        -------
        float or numpy.ndarray
            The derivative of the sigmoid function evaluated at the given input(s). 
            If the input is a scalar, the output will be a scalar; if the input is an 
            array, the output will be an array of the same shape with the derivative 
            applied element-wise.
        """
        return x * (1 - x)
    
    def _mse(self, target, output):
        """
        Compute the Mean Squared Error (MSE) between the target and output values.

        The Mean Squared Error is a common loss function used in regression tasks and 
        machine learning models. It calculates the average of the squared differences 
        between the predicted outputs and the actual target values. A lower MSE indicates 
        better model performance.

        MSE is defined as: 
        MSE = (1/n) * Σ(target_i - output_i)², where `n` is the number of data points.

        Parameters:
        ----------
        target : numpy.ndarray or list
            The actual target values that the model is trying to predict. It should have 
            the same shape as `output`.

        output : numpy.ndarray or list
            The predicted values from the model. It should have the same shape as `target`.

        Returns:
        -------
        float
            The Mean Squared Error between the target and output values. It is a scalar value.
    """
        return np.average( (target - output) ** 2 )
    
    def gradient_descent(self, learnin_rate=1):
        """
        Perform one step of gradient descent to update the model's weights.

        Gradient descent is an optimization algorithm used to minimize the loss function
        by iteratively adjusting the weights in the direction of the negative gradient 
        (derivatives). This function updates the weights by moving in the direction that 
        reduces the error based on the derivatives of the loss function with respect to 
        the weights. The size of the update is controlled by the learning rate.

        Parameters:
        ----------
        learning_rate : float, optional, default=1
            The learning rate controls the step size during the weight update. A higher value 
            may lead to faster convergence but risks overshooting the minimum, while a 
            smaller value leads to slower convergence but more stable updates.

        Returns:
        -------
        None
            This function modifies the weights of the model in place and does not return any value.
        """
        for i in range( len(self.weights) ):
            weights = self.weights[i]
            derivatives = self.derivatives[i]
            weights += derivatives * learnin_rate

    def foward_propagate(self, inputs):
        """
        Perform forward propagation through the neural network.

        Forward propagation computes the output of the network by passing the input 
        through each layer, applying the weights, and using an activation function 
        (typically sigmoid). It updates the activations of each layer and stores them 
        for use in backpropagation.

        During forward propagation:
        - The input is multiplied by the weights of each layer.
        - The activation function (sigmoid) is applied to the resulting values.
        - The activations are stored for backpropagation to calculate gradients.

        Parameters:
        ----------
        inputs : numpy.ndarray
            The input data to the network. It is passed through the network's layers 
            to compute the output.

        Returns:
        -------
        numpy.ndarray
            The final output of the network after passing through all layers. 
            This is the result of applying the activation function to the final 
            layer's weighted input.
        """
        activations = inputs

        # save the activations for backpropagation
        self.activations[0] = activations

        # iterate through the net works layers
        for idx, w in enumerate(self.weights):
            # calculate the inputs
            # Matrix multiplication between previous activation and weight matrix
            net_inputs = np.dot(activations, w)

            # calculate sigmoid activation function
            activations = self._sigmoid(net_inputs)

            # save activation for backpropagation
            self.activations[idx + 1] = activations

        return activations

    def back_propagate(self, error):
        """
        Perform backpropagation to compute the gradients of the loss function with respect to the weights.

        Backpropagation is a key algorithm used to update the weights in a neural network. 
        It calculates the gradients (derivatives) of the loss function with respect to each 
        weight by propagating the error backward through the network. These gradients are 
        used to adjust the weights during training to minimize the error.

        The process involves:
        - Computing the delta (error term) for each layer.
        - Calculating the gradients of the weights by multiplying the delta with the 
        activations from the previous layer.
        - Updating the error term to propagate to the previous layer.

        Parameters:
        ----------
        error : numpy.ndarray
            The error term (difference between target and output) that will be propagated 
            backward through the network to calculate the gradients.

        Returns:
        -------
        None
            This function updates the `derivatives` and `error` attributes in place, which are 
            later used for adjusting the weights during gradient descent.
        """
        for i in reversed(range(len(self.derivatives))):
            activations = self.activations[i+1]

            delta = error * self._sigmoid_derivative( activations )

            delta_re = delta.reshape( delta.shape[0], -1 ).T

            current_activations = self.activations[i]

            current_activations = current_activations.reshape( current_activations.shape[0], -1 )

            self.derivatives[i] = np.dot( current_activations, delta_re )

            error = np.dot( delta, self.weights[i].T )
    
    def train(self, inputs, targets, epochs, learning_rate):
        """
        Train the neural network using forward propagation, backpropagation, and gradient descent.

        This function trains the model over a specified number of epochs by iteratively
        performing forward propagation, calculating the error, applying backpropagation
        to compute gradients, and updating the weights using gradient descent. The error 
        for each training example is computed using Mean Squared Error (MSE).

        After each epoch, the average error is reported to give an indication of the 
        model's performance and progress.

        Parameters:
        ----------
        inputs : numpy.ndarray
            The input data used to train the model. Each row represents one input example.

        targets : numpy.ndarray
            The target labels corresponding to the input data. Each row represents the 
            target for the corresponding input.

        epochs : int
            The number of times the training process will iterate over the entire dataset.

        learning_rate : float
            The learning rate controls the size of the weight updates during gradient descent.
        
        Returns:
        -------
        None
            The function modifies the model's weights in place and does not return any value.
        """
        for i in range( epochs ):
            sum_errors = 0
            for j, input in enumerate( inputs ):
                target = targets[j]

                output = self.foward_propagate(input)

                error = target - output

                self.back_propagate( error )

                self.gradient_descent( learning_rate )

                sum_errors += self._mse(target, output)

            # Epoch complete, report the training error
            print("Error: {} at epoch {}".format(sum_errors / len(inputs), i+1))

        print("Training complete!")
        print("=====")

    def generate_c_code(self):
        """
        Generate the C the files `uMLP.h`, and `uMLP.c` for a neural network with weights, activation functions, 
        and matrix operations.

        This function generates C code that defines the structure and operations necessary to represent 
        and manipulate the weights of a multi-layer perceptron (MLP) neural network. The header file 
        (`uMLP.h`) which includes:
        - Structure definitions for the network's weight matrices.
        - Matrix multiplication function (`uMLP_mult_matrix`).
        - Sigmoid activation function (`sigmoide`).
        - Functions for initialization (`uMLP_InitMatrix`) and deallocation (`uMLP_FreeMatrix`) of weight matrices.
        - A function to predict output (`uMLP_Predict`) based on the input.

        The generated C code is designed for managing MLP weights and performing basic matrix operations.

        Returns:
        -------
        None
            The function writes the generated C code into `uMLP.h`, and `uMLP.c` file. It does not return anything.
        """

        struct_matriz = """
struct uMLPWeights
{"""
        for w in range(len(self.weights)):
            struct_matriz += "\n\tdouble **m{};".format(w)

        struct_matriz += "\n"
        for w in range(len(self.weights)):
            struct_matriz += "\n\tdouble m{}_columns;".format(w)
            struct_matriz += "\n\tdouble m{}_rows;".format(w)
        
        struct_matriz += "\n};"

        f = open("uMLP.h", "w")
        text = """#ifndef UMLP_H
#define UMLP_H

#include <stdlib.h>

""" + str(struct_matriz) + f"""

#define INPUT_ROWS {1}
#define INPUT_COLUMNS {self.num_inputs}

void uMLP_InitMatrix(struct uMLPWeights *layers);

void uMLP_FreeMatrix(struct uMLPWeights *layers);

double **uMLP_mult_matrix(double **a, int rows_in_a, int columns_in_a, double **b, int rows_in_b, int columns_in_b);

void sigmoide(double **a, int number_of_elements);

void uMLP_Predict(double **input, double *out, struct uMLPWeights layers);

#endif
        """
        f.write(text)
        f.close()

        c_file = open("uMLP.c", "w")
        text = """#include "uMLP.h"
#include <math.h>

void uMLP_InitMatrix(struct uMLPWeights *layers)
{"""

        for w in range(len(self.weights)):
            number_of_rows = len(self.weights[w])
            number_of_columns = len(self.weights[w][0])
            mem_aloc = f"""
    layers->m{w} = (double **)malloc({number_of_rows} * sizeof(double *));
    for (int i = 0; i < {number_of_rows}; i++)
"""
            mem_aloc += "\t{"
            mem_aloc += f"""
        layers->m{w}[i] = (double *)malloc({number_of_columns} * sizeof(double));
"""
            mem_aloc += "\t}\n"
            mem_aloc += f"""
    layers->m{w}_columns = {number_of_columns};
    layers->m{w}_rows = {number_of_rows};
\n"""

            text += mem_aloc

            for r in range(number_of_rows):
                for c in range(number_of_columns):
                    text += f"\tlayers->m{w}[{r}][{c}] = {self.weights[w][r][c]};\n"

        text += "}\n\n"

        text += """void uMLP_FreeMatrix(struct uMLPWeights *layers)
{\n"""
        for w in range(len(self.weights)):
            text += "\tfor (int i = 0; i < {}; i++)\n".format(len(self.weights[w]))
            text += "\t{\n"
            text += "\t\tfree(layers->m{}[i]);\n".format(w)
            text += "\t}\n"
            text += "\tfree(layers->m{});\n".format(w)

        text += "}\n\n"

        text += """void uMLP_FreeTargetMatrix(double **matrix, int rows)
{
	for (int i = 0; i < rows; i++)
    {
		free(matrix[i]);
    }
	free(matrix);
}\n\n"""

        text += """double **uMLP_mult_matrix(double **a, int rows_in_a, int columns_in_a, double **b, int rows_in_b, int columns_in_b)
{
    if (columns_in_a != rows_in_b)
    {
        return NULL;
    }

    double **dot = (double **)malloc(rows_in_a * sizeof(double *));
    for (int i = 0; i < rows_in_a; i++)
    {
        dot[i] = (double *)calloc(columns_in_b, sizeof(double));
    }

    for (int i = 0; i < rows_in_a; i++)
    {
        for (int j = 0; j < columns_in_b; j++)
        {
            for (int k = 0; k < columns_in_a; k++)
            {
                dot[i][j] += a[i][k] * b[k][j];
            }
        }
    }

    return dot;
}

void sigmoide(double **current_layer_values, int number_of_elements)
{
	for (int i = 0; i < number_of_elements; i++)
	{
		current_layer_values[0][i] = 1.0 / (1.0 + exp(-current_layer_values[0][i]));
	}
}\n\n"""

        text += """void uMLP_Predict(double **input, double *out, struct uMLPWeights layers)
{
    double **result_m0 = uMLP_mult_matrix(input, INPUT_ROWS, INPUT_COLUMNS, layers.m0, layers.m0_rows, layers.m0_columns);
	sigmoide(result_m0, layers.m0_columns);
"""
        
        for i in range(1, len(self.weights)):
            text += """
    double **result_m{} = uMLP_mult_matrix(result_m{}, INPUT_ROWS, layers.m{}_columns, layers.m{}, layers.m{}_rows, layers.m{}_columns);
	uMLP_FreeTargetMatrix(result_m{}, INPUT_ROWS);
    sigmoide(result_m{}, layers.m{}_columns);
""".format(i, i - 1 , i-1, i, i, i, i - 1, i, i)

        text += """
    *out = result_m{}[0][0];
	uMLP_FreeTargetMatrix(result_m{}, INPUT_ROWS);
""".format(len(self.weights) - 1, len(self.weights) - 1)

        text += "}\n\n"

        c_file.write(text)
        c_file.close()

if __name__ == "__main__":
    items = np.array([[random()/2 for _ in range(2)] for _ in range(1000)])
    targets = np.array([[i[0] + i[1]] for i in items])

    mlp = MLP(2, 1, [3, 5])

    mlp.train(items, targets, 150, 0.1)

    # predict
    input = np.array([0.3, 0.1])
    outputs = mlp.foward_propagate(input)

    print(f"{input = }")
    print(f"{outputs = }")