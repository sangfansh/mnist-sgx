/*
 * Copyright (C) 2011-2018 Intel Corporation. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in
 *     the documentation and/or other materials provided with the
 *     distribution.
 *   * Neither the name of Intel Corporation nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */


#include <stdarg.h>
#include <stdio.h>      /* vsnprintf */
#include <stdlib.h>

#include "sgx_trts.h"
#include "Enclave.h"
#include "sgx_tseal.h"
#include "sealing/sealing.h"
#include "Enclave_t.h"  /* print_string */

#include "math.h"
#include "../App/App.h"

// NN described in array

// Number of training samples
const int nTraining = 60000;

// Image size in MNIST database
const int width = 28;
const int height = 28;

// n1 = Number of input neurons
// n2 = Number of hidden neurons
// n3 = Number of output neurons
// epochs = Number of iterations for back-propagation algorithm
// learning_rate = Learing rate
// momentum = Momentum (heuristics to optimize back-propagation algorithm)
// epsilon = Epsilon, no more iterations if the learning error is smaller than epsilon

const int n1 = width * height; // = 784, without bias neuron
const int n2 = 128;
const int n3 = 10; // Ten classes: 0 - 9
const int epochs = 512;
const double learning_rate = 1e-3;
const double momentum = 0.9;
const double epsilon = 1e-3;

NeuralNetwork *nn;
double test[3];

// From layer 1 to layer 2. Or: Input layer - Hidden layer
double delta1[785][129], out1[785];

// From layer 2 to layer 3. Or; Hidden layer - Output layer
double delta2[129][11], in2[129], out2[129], theta2[129];

// Layer 3 - Output layer
double in3[11], out3[11], theta3[11];
double expected[11];


// Image. In MNIST: 28x28 gray scale images.
int d[width + 1][height + 1];

void init_array()
{
    uint32_t val;
    // sgx_read_rand((unsigned char *) &val, 4);
    nn = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    test[0] = 0.1;
    test[1] = 1.1;
    test[2] = 2.1;

    // Initialization for weights from Input layer to Hidden layer
    for (int i = 1; i <= n1; ++i) {
        for (int j = 1; j <= n2; ++j) {
            sgx_read_rand((unsigned char *) &val, 4);
            int sign = val % 2;

            sgx_read_rand((unsigned char *) &val, 4);
            nn->w1[i][j] = (double)(val % 6) / 10.0;
            if (sign == 1) {
                nn->w1[i][j] = - nn->w1[i][j];
            }
        }
    }

    // Initialization for weights from Hidden layer to Output layer
    for (int i = 1; i <= n2; ++i) {
        for (int j = 1; j <= n3; ++j) {
            sgx_read_rand((unsigned char *) &val, 4);
            int sign = val % 2;

            sgx_read_rand((unsigned char *) &val, 4);
            nn->w2[i][j] = (double)(val % 10 + 1) / (10.0 * n3);
            if (sign == 1) {
                nn->w2[i][j] = - nn->w2[i][j];
            }
        }
    }
}

// sigmoid function
double sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-x));
}

// Forward process - perceptron
void perceptron()
{
    for (int i = 1; i <= n2; ++i) {
        in2[i] = 0.0;
    }

    for (int i = 1; i <= n3; ++i) {
        in3[i] = 0.0;
    }

    for (int i = 1; i <= n1; ++i) {
        for (int j = 1; j <= n2; ++j) {
            in2[j] += out1[i] * nn->w1[i][j];
        }
    }

    for (int i = 1; i <= n2; ++i) {
        out2[i] = sigmoid(in2[i]);
    }

    for (int i = 1; i <= n2; ++i) {
        for (int j = 1; j <= n3; ++j) {
            in3[j] += out2[i] * nn->w2[i][j];
        }
    }

    for (int i = 1; i <= n3; ++i) {
        out3[i] = sigmoid(in3[i]);
    }
}

// Norm L2 error
double square_error()
{
    double res = 0.0;
    for (int i = 1; i <= n3; ++i) {
        res += (out3[i] - expected[i]) * (out3[i] - expected[i]);
    }
    res *= 0.5;
    return res;
}

// Back Propagation Algorithm
void back_propagation()
{
    double sum;

    for (int i = 1; i <= n3; ++i) {
        theta3[i] = out3[i] * (1 - out3[i]) * (expected[i] - out3[i]);
    }

    for (int i = 1; i <= n2; ++i) {
        sum = 0.0;
        for (int j = 1; j <= n3; ++j) {
            sum += nn->w2[i][j] * theta3[j];
        }
        theta2[i] = out2[i] * (1 - out2[i]) * sum;
    }

    for (int i = 1; i <= n2; ++i) {
        for (int j = 1; j <= n3; ++j) {
            delta2[i][j] = (learning_rate * theta3[j] * out2[i]) + (momentum * delta2[i][j]);
            nn->w2[i][j] += delta2[i][j];
        }
    }

    for (int i = 1; i <= n1; ++i) {
        for (int j = 1 ; j <= n2 ; j++ ) {
            delta1[i][j] = (learning_rate * theta2[j] * out1[i]) + (momentum * delta1[i][j]);
            nn->w1[i][j] += delta1[i][j];
        }
    }
}

// Learning process: perceptron - back propagation
int learning_process()
{
    for (int i = 1; i <= n1; ++i) {
        for (int j = 1; j <= n2; ++j) {
            delta1[i][j] = 0.0;
        }
    }

    for (int i = 1; i <= n2; ++i) {
        for (int j = 1; j <= n3; ++j) {
            delta2[i][j] = 0.0;
        }
    }

    for (int i = 1; i <= epochs; ++i) {
        perceptron();
        back_propagation();
        if (square_error() < epsilon) {
            return i;
        }
    }
    return epochs;
}

// reading input - gray scale image and the corresponding label
int input()
{
    // reading image
    char number;
    for (int j = 1; j <= height; ++j) {
        for (int i = 1; i <= width; ++i) {
            ocall_read_image(&number);
            if (number == 0) {
                d[i][j] = 0;
            }
            else {
                d[i][j] = 1;
            }
        }
    }

    /*
    printf("Image:\n");
    for (int j = 1; j <= height; ++j) {
        for (int i = 1; i <= width; ++i) {
            printf("%d", d[i][j]);
        }
        printf("\n");
    }
    */

    for (int j = 1; j <= height; ++j) {
        for (int i = 1; i <= width; ++i) {
            int pos = i + (j - 1) * width;
            out1[pos] = d[i][j];
        }
    }

    // reading label
    ocall_read_label(&number);
    for (int i = 1; i <= n3; ++i) {
        expected[i] = 0.0;
    }
    expected[number + 1] = 1.0;

    // printf("Label: %d\n", (int)number);
    return (int)(number);
}

void save_nn() {
    sgx_status_t ocall_status, sealing_status;
    int ocall_ret;
    size_t sealed_size = sizeof(sgx_sealed_data_t) + sizeof(NeuralNetwork);
    printf("size of nn is %d\n", sizeof(NeuralNetwork));
    uint8_t* sealed_data = (uint8_t*)malloc(sealed_size);
    sealing_status = seal_nn(nn, (sgx_sealed_data_t*)sealed_data, sealed_size);
    free(nn);
    if (sealing_status != SGX_SUCCESS) {
        free(sealed_data);
        printf("Sealing NN failed!\n");
        print_error_message(sealing_status);
        return;
    }

    ocall_status = ocall_save_nn(&ocall_ret, sealed_data, sealed_size);
    free(sealed_data);
    if (ocall_ret != 0 || ocall_status != SGX_SUCCESS) {
        printf("Saving NN failed!");
        return;
    }
}

void load_model() {
    sgx_status_t ocall_status, sealing_status;
    int ocall_ret;

    size_t sealed_size = sizeof(sgx_sealed_data_t) + sizeof(NeuralNetwork);
    uint8_t* sealed_data = (uint8_t*)malloc(sealed_size);
    ocall_status = ocall_load_nn(&ocall_ret, sealed_data, sealed_size);
    if (ocall_ret != 0 || ocall_status != SGX_SUCCESS) {
        free(sealed_data);
        printf("Loading NN failed!\n");
    }

    uint32_t plaintext_size = sizeof(NeuralNetwork);
    nn = (NeuralNetwork*)malloc(plaintext_size);
    sealing_status = unseal_nn((sgx_sealed_data_t*)sealed_data, nn, plaintext_size);
    free(sealed_data);
    if (sealing_status != SGX_SUCCESS) {
        free(nn);
        printf("Unsealing NN failed!\n");
    }

}

int predict() {
    perceptron();
    int result = 1;
    for (int i = 2; i <= n3; ++i) {
        if (out3[i] > out3[result]) {
            result = i;
        }
    }
    --result;

    return result;
}

/*
 * printf:
 *   Invokes OCALL to display the enclave buffer to the terminal.
 */
void printf(const char *fmt, ...)
{
    char buf[BUFSIZ] = {'\0'};
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, BUFSIZ, fmt, ap);
    va_end(ap);
    ocall_print_string(buf);
}

void printf_helloworld()
{
    printf("Hello World\n");
}

