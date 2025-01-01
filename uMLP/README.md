# MLP to C Code Generator

This project is a Python implementation of a Multi-Layer Perceptron (MLP) designed for training and generating equivalent C code. The generated C code can be deployed on microcontrollers, making it ideal for embedded system applications that require neural network predictions. 

NOTICE: All examples will demonstrate an MLP that predicts the sum of 2 numbers.

## Features

- **Train an MLP**: Supports multiple layers and neurons with customizable input, hidden, and output layers.
- **Forward Propagation**: Calculates the output for a given input using the sigmoid activation function.
- **Backpropagation**: Updates weights based on the error gradient to optimize the network.
- **Custom Loss Function**: Implements Mean Squared Error (MSE) as the loss metric.
- **C Code Generation**: Automatically generates `uMLP.h` and `uMLP.c` files for deploying the trained network in C environments.

## Prerequisites

- Python 3.x
- NumPy
## How to Use

### Train and Generate the C code

1. **Train the MLP**:
   - Initialize the MLP with desired parameters (e.g., number of inputs, hidden layers, and outputs).
   - Train the network using your dataset.

2. **Generate C Code**:

    - After training, call the generate_c_code method to generate uMLP.h and uMLP.c.

    Example:
    ```python
    from uMLP import MLP
    import numpy as np
    from random import random

    # Generate synthetic data
    items = np.array([[random()/2 for _ in range(2)] for _ in range(1000)])
    targets = np.array([[i[0] + i[1]] for i in items])

    # Initialize and train the MLP
    mlp = MLP(2, 1, [3, 5])
    mlp.train(items, targets, epochs=150, learning_rate=0.1)
    mlp.generate_c_code()
    ```

### Execute The C Code With gcc

Here is an example of how you can run your code and compile it with gcc:

Example(main.c):
```C
#include <stdio.h>
#include <stdlib.h>
#include "uMLP.c"

int main()
{
    double **input = (double **)malloc(1 * sizeof(double *));
    input[0] = (double *)malloc(2 * sizeof(double));
    input[0][0] = 0.1;
    input[0][1] = 0.5;
    double out = 0;

    struct uMLPWeights layers;

    uMLP_InitMatrix(&layers);
    uMLP_Predict(input, &out, layers);
    uMLP_FreeMatrix(&layers);

    printf("Predicted Result: %f\n", out);

    return 0;
}
```

### Use The Code Inside a Microcontroller

In this tutorial, we will explain how to use the output files inside an ESP32, Arduino, or STM32 project.

#### STM32

1. Create a new STM32 project.
2. Copy `uMLP.c` into the `Core\Src` directory.
3. Copy `uMLP.h` into the `Core\Inc` directory.
4. Add the following code to `Core\Src\main.c`:

Example STM32(main.c):
```C
...

/* USER CODE BEGIN Includes */
#include "uMLP.h"
/* USER CODE END Includes */

...

/* USER CODE BEGIN 2 */
double **input = (double **)malloc(1 * sizeof(double *));
input[0] = (double *)malloc(2 * sizeof(double));

double out = 0.0;
struct uMLPWeights layers;

uMLP_InitMatrix(&layers);
/* USER CODE END 2 */

...

while (1)
{
...

/* USER CODE BEGIN 3 */
    input[0][0] = 0.1;
    input[0][1] = 0.5;
    uMLP_Predict(input, &out, layers);
/* USER CODE END 3 */
```

The result will be stored in the `out` variable.

#### ESP32 or Arduino

To use the uMLP inside your ESP32 or Arduino project:

1. Copy `uMLP.c` and `uMLP.h` into your project directory.
2. Add the following code to your main file:

Example:
```C
#include "uMLP.h"

double **input;
double out;
struct uMLPWeights layers;

void setup()
{
    uMLP_InitMatrix(&layers);
    input = (double **)malloc(1 * sizeof(double *));
    input[0] = (double *)malloc(2 * sizeof(double));
}

void loop()
{
    input[0][0] = 0.1;
    input[0][1] = 0.5;
    uMLP_Predict(input, &out, layers);
}
```

The result will be stored in the `out` variable.


## File Structure

- **uMLP.py**: Core implementation of the MLP and code generation logic.
- **uMLP.h** & **uMLP.c**: Generated files for deploying the trained network in C.

## Applications

This project is useful for deploying machine learning models on:

- Embedded systems
- IoT devices
- Low-power hardware where Python is not feasible

## TODO and Performance

- Matrix multiplication has the associative property, which means that for matrices $A_{m \times n}$, $B_{n \times o}$, and $C_{o \times q}$, the final product of these matrices can be mathematically represented as shown in Equation 1. With this in mind, you might wonder, "Why are there so many matrices in the uMLP code?" Remember, each layer in the MLP has its own matrix. The reason for this is simple: to keep the code readable and easy to understand. If you'd like to improve the code and reduce memory usage, it's possible to multiply matrices sequentially—multiply $m_0$ through $m_n$, where $n$ is the last weight matrix in the uMLPWeights array, and obtain a final matrix that can be used to multiply the input.

<center>

$(AB)C = A(BC) \ \ \ \ \ \ \ \ (1)$

</center>

- Each matrix inside `uMLPWeights` has an associated variable indicating the number of rows and columns it contains. We cannot use `sizeof(matrix)` (at least, I did not) because memory is allocated dynamically to create all the matrices.

- Up to this point, the uMLP can only receive input of type $input_{1, m}$. This feature is intended to be improved over time.

## Contact

If you have any suggestions or questions about how it works, feel free to send me an email.
