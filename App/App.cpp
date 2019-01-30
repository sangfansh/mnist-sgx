/*
 * Copyright (C) 2011-2018 Intel Corporation. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
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


#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <iostream>
#include <fstream>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <set>
#include <iterator>
#include <algorithm>

using namespace std;

# include <unistd.h>
# include <pwd.h>
# define MAX_PATH FILENAME_MAX

#include "sgx_urts.h"
#include "App.h"
#include "Enclave_u.h"

#define BUFLEN 2048
#define SGX_AESGCM_MAC_SIZE 16
#define SGX_AESGCM_IV_SIZE 12

// Training image file name
const string training_image_fn = "mnist/train-images.idx3-ubyte";

// Training label file name
const string training_label_fn = "mnist/train-labels.idx1-ubyte";

// Testing image file name
const string testing_image_fn = "mnist/t10k-images.idx3-ubyte";

// Testing label file name
const string testing_label_fn = "mnist/t10k-labels.idx1-ubyte";

// Weights file name
const string model_fn = "model-neural-network.dat";

// Training Report file name
const string training_report_fn = "training-report.dat";

// Testing report file name
const string testing_report_fn = "testing-report.dat";

// Number of training samples
const int nTraining = 10000;

// Number of testing samples
const int nTesting = 10000;

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

// From layer 1 to layer 2. Or: Input layer - Hidden layer
// double *w1[n1 + 1], *delta1[n1 + 1], *out1;

// From layer 2 to layer 3. Or; Hidden layer - Output layer
// double *w2[n2 + 1], *delta2[n2 + 1], *in2, *out2, *theta2;

// Layer 3 - Output layer
// double *in3, *out3, *theta3;
// double expected[n3 + 1];

// Image. In MNIST: 28x28 gray scale images.
// int d[width + 1][height + 1];

// File stream to read data (image, label) and write down a report
ifstream image;
ifstream label;
ofstream report;

// Software details
void training_about() {
    cout << "**************************************************" << endl;
    cout << "*** Training Neural Network for MNIST database ***" << endl;
    cout << "**************************************************" << endl;
    cout << endl;
    cout << "No. input neurons: " << n1 << endl;
    cout << "No. hidden neurons: " << n2 << endl;
    cout << "No. output neurons: " << n3 << endl;
    cout << endl;
    cout << "No. iterations: " << epochs << endl;
    cout << "Learning rate: " << learning_rate << endl;
    cout << "Momentum: " << momentum << endl;
    cout << "Epsilon: " << epsilon << endl;
    cout << endl;
    cout << "Training image data: " << training_image_fn << endl;
    cout << "Training label data: " << training_label_fn << endl;
    cout << "No. training sample: " << nTraining << endl << endl;
}

void testing_about() {
    // Details
    cout << "*************************************************" << endl;
    cout << "*** Testing Neural Network for MNIST database ***" << endl;
    cout << "*************************************************" << endl;
    cout << endl;
    cout << "No. input neurons: " << n1 << endl;
    cout << "No. hidden neurons: " << n2 << endl;
    cout << "No. output neurons: " << n3 << endl;
    cout << endl;
    cout << "Testing image data: " << testing_image_fn << endl;
    cout << "Testing label data: " << testing_label_fn << endl;
    cout << "No. testing sample: " << nTesting << endl << endl;
}

/* Global EID shared by multiple threads */
sgx_enclave_id_t global_eid = 0;

typedef struct _sgx_errlist_t {
    sgx_status_t err;
    const char *msg;
    const char *sug; /* Suggestion */
} sgx_errlist_t;

/* Error code returned by sgx_create_enclave */
static sgx_errlist_t sgx_errlist[] = {
    {
        SGX_ERROR_UNEXPECTED,
        "Unexpected error occurred.",
        NULL
    },
    {
        SGX_ERROR_INVALID_PARAMETER,
        "Invalid parameter.",
        NULL
    },
    {
        SGX_ERROR_OUT_OF_MEMORY,
        "Out of memory.",
        NULL
    },
    {
        SGX_ERROR_ENCLAVE_LOST,
        "Power transition occurred.",
        "Please refer to the sample \"PowerTransition\" for details."
    },
    {
        SGX_ERROR_INVALID_ENCLAVE,
        "Invalid enclave image.",
        NULL
    },
    {
        SGX_ERROR_INVALID_ENCLAVE_ID,
        "Invalid enclave identification.",
        NULL
    },
    {
        SGX_ERROR_INVALID_SIGNATURE,
        "Invalid enclave signature.",
        NULL
    },
    {
        SGX_ERROR_OUT_OF_EPC,
        "Out of EPC memory.",
        NULL
    },
    {
        SGX_ERROR_NO_DEVICE,
        "Invalid SGX device.",
        "Please make sure SGX module is enabled in the BIOS, and install SGX driver afterwards."
    },
    {
        SGX_ERROR_MEMORY_MAP_CONFLICT,
        "Memory map conflicted.",
        NULL
    },
    {
        SGX_ERROR_INVALID_METADATA,
        "Invalid enclave metadata.",
        NULL
    },
    {
        SGX_ERROR_DEVICE_BUSY,
        "SGX device was busy.",
        NULL
    },
    {
        SGX_ERROR_INVALID_VERSION,
        "Enclave version was invalid.",
        NULL
    },
    {
        SGX_ERROR_INVALID_ATTRIBUTE,
        "Enclave was not authorized.",
        NULL
    },
    {
        SGX_ERROR_ENCLAVE_FILE_ACCESS,
        "Can't open enclave file.",
        NULL
    },
};

/* Check error conditions for loading enclave */
void print_error_message(sgx_status_t ret)
{
    size_t idx = 0;
    size_t ttl = sizeof sgx_errlist/sizeof sgx_errlist[0];

    for (idx = 0; idx < ttl; idx++) {
        if(ret == sgx_errlist[idx].err) {
            if(NULL != sgx_errlist[idx].sug)
                printf("Info: %s\n", sgx_errlist[idx].sug);
            printf("Error: %s\n", sgx_errlist[idx].msg);
            break;
        }
    }

    if (idx == ttl)
    	printf("Error code is 0x%X. Please refer to the \"Intel SGX SDK Developer Reference\" for more details.\n", ret);
}

/* Initialize the enclave:
 *   Step 1: try to retrieve the launch token saved by last transaction
 *   Step 2: call sgx_create_enclave to initialize an enclave instance
 *   Step 3: save the launch token if it is updated
 */
int initialize_enclave(void)
{
    char token_path[MAX_PATH] = {'\0'};
    sgx_launch_token_t token = {0};
    sgx_status_t ret = SGX_ERROR_UNEXPECTED;
    int updated = 0;

    /* Step 1: try to retrieve the launch token saved by last transaction
     *         if there is no token, then create a new one.
     */
    /* try to get the token saved in $HOME */
    const char *home_dir = getpwuid(getuid())->pw_dir;

    if (home_dir != NULL &&
        (strlen(home_dir)+strlen("/")+sizeof(TOKEN_FILENAME)+1) <= MAX_PATH) {
        /* compose the token path */
        strncpy(token_path, home_dir, strlen(home_dir));
        strncat(token_path, "/", strlen("/"));
        strncat(token_path, TOKEN_FILENAME, sizeof(TOKEN_FILENAME)+1);
    } else {
        /* if token path is too long or $HOME is NULL */
        strncpy(token_path, TOKEN_FILENAME, sizeof(TOKEN_FILENAME));
    }

    FILE *fp = fopen(token_path, "rb");
    if (fp == NULL && (fp = fopen(token_path, "wb")) == NULL) {
        printf("Warning: Failed to create/open the launch token file \"%s\".\n", token_path);
    }

    if (fp != NULL) {
        /* read the token from saved file */
        size_t read_num = fread(token, 1, sizeof(sgx_launch_token_t), fp);
        if (read_num != 0 && read_num != sizeof(sgx_launch_token_t)) {
            /* if token is invalid, clear the buffer */
            memset(&token, 0x0, sizeof(sgx_launch_token_t));
            printf("Warning: Invalid launch token read from \"%s\".\n", token_path);
        }
    }
    /* Step 2: call sgx_create_enclave to initialize an enclave instance */
    /* Debug Support: set 2nd parameter to 1 */
    ret = sgx_create_enclave(ENCLAVE_FILENAME, SGX_DEBUG_FLAG, &token, &updated, &global_eid, NULL);
    if (ret != SGX_SUCCESS) {
        print_error_message(ret);
        if (fp != NULL) fclose(fp);
        return -1;
    }

    /* Step 3: save the launch token if it is updated */
    if (updated == FALSE || fp == NULL) {
        /* if the token is not updated, or file handler is invalid, do not perform saving */
        if (fp != NULL) fclose(fp);
        return 0;
    }

    /* reopen the file with write capablity */
    fp = freopen(token_path, "wb", fp);
    if (fp == NULL) return 0;
    size_t write_num = fwrite(token, 1, sizeof(sgx_launch_token_t), fp);
    if (write_num != sizeof(sgx_launch_token_t))
        printf("Warning: Failed to save launch token to \"%s\".\n", token_path);
    fclose(fp);
    return 0;
}

/* OCall functions */
void ocall_print_string(const char *str)
{
    /* Proxy/Bridge will check the length and null-terminate
     * the input string to prevent buffer overflow.
     */
    printf("%s", str);
}

void ocall_print_num(uint32_t num)
{
    cout << num << endl;
}

void ocall_read_image(char *number)
{
    image.read(number, sizeof(char));
}

void ocall_read_label(char *number)
{
    label.read(number, sizeof(char));
}

void ocall_read_one(char *data)
{
    //cout << "In ocall_read_one\n";
    //char *result = (char *)malloc(785 * sizeof(char));
    char result[786];
    image.read(result, 784 * sizeof(char));
    label.read(&result[784], sizeof(char));
    result[785] = 0;
    //for(int i = 0; i < 784; i++)
    //{
    //    image.read(result + i, sizeof(char));
    //}
    //label.read(result + 784, sizeof(char));
    //cout << "After read image and label\n";
    int cnt = 0;
    cout << "Image:\n";
    for(int j = 1; j <= height; ++j) {
        for(int i = 1; i <= width; ++i) {
            if(result[cnt] == 0) {
                cout << "0";
                result[cnt] = 0;
            }
            else {
                cout << "1";
                result[cnt] = 1;
            }
            cnt++;
        }
        cout << endl;
    }
    cout << "Label: " << (int)result[784] << endl;


    sgx_status_t ret;

    //printf("Result: %s\n", result);
    size_t encMessageLen = (SGX_AESGCM_MAC_SIZE + SGX_AESGCM_IV_SIZE + 785);
    char *encMessage = (char *) malloc((encMessageLen+1)*sizeof(char));
    printf("Encrypting...\n");
    ret = encryptMessage(global_eid, result, 785, encMessage, encMessageLen);
    encMessage[encMessageLen] = '\0';
    printf("Encrypted image data: %s\n", encMessage);
    //size_t decMessageLen = strlen(result);
    //char *decMessage = (char *) malloc((decMessageLen+1)*sizeof(char));
    //printf("Decrypting...\n");
    //ret = decryptMessage(global_eid,encMessage,encMessageLen,decMessage,decMessageLen);
    //decMessage[decMessageLen] = '\0';
    //printf("Decrypted message: %s\n", decMessage);


    /*
    printf("result:\n");
    printf("%s\n", result);
    //size_t encMessageLen = (SGX_AESGCM_MAC_SIZE + SGX_AESGCM_IV_SIZE + 785);
    encMessageLen = (SGX_AESGCM_MAC_SIZE + SGX_AESGCM_IV_SIZE + 785);
    char *message = (char *) malloc((encMessageLen+1)*sizeof(char));
    printf("Size to encrypt: %d\n", encMessageLen);
    printf("strlen(result): %d\n", strlen(result2));
    ret = encryptMessage(global_eid, (char *)result2, strlen(result2), message, encMessageLen);
    cout << "After encryptMessage\n";
    message[encMessageLen] = '\0';
    cout << "Shouldn't print out enclave content\n";
    printf("Encrypted message: %s\n", message);
    printf("Encrypted size: %d\n", strlen(message));

    //size_t decMessageLen = strlen(message);
    decMessageLen = strlen(message);
    //char *decMessage = (char *) malloc((decMessageLen+1)*sizeof(char));
    decMessage = (char *) malloc((decMessageLen+1)*sizeof(char));
    printf("Decrypting...\n");
    ret = decryptMessage(global_eid,message,encMessageLen,decMessage,decMessageLen);
    printf("Decrypted message: %s", decMessage);
    */

    memcpy(data, encMessage, encMessageLen);

    free(encMessage);
    //free(result);
}

void input_test() {
    char number;
    cout << "Image:" << endl;
    for (int j = 1; j <= height; ++j) {
        for (int i = 1; i <= width; ++i) {
            image.read(&number, sizeof(char));
            if (number == 0) {
                cout << 0;
            } else {
                cout << 1;
            }
        }
        cout << endl;
    }

    label.read(&number, sizeof(char));
    cout << "Label: " << (int)(number) << endl;
}

int ocall_save_nn(const uint8_t* sealed_data, const size_t sealed_size) {
    ofstream file(model_fn, ios::out | ios::binary);
    if (file.fail()) {return 1;}
    file.write((const char*) sealed_data, sealed_size);
    file.close();
    return 0;
}

int ocall_load_nn(uint8_t* sealed_data, const size_t sealed_size) {
    ifstream file(model_fn, ios::in | ios::binary);
    if (file.fail()) {return 1;}
    file.read((char*) sealed_data, sealed_size);
    file.close();
    return 0;

}

/* Application entry */
int SGX_CDECL main(int argc, char *argv[])
{
    (void)(argc);
    (void)(argv);

    /* Initialize the enclave */
    if(initialize_enclave() < 0){
        printf("Enter a character before exit ...\n");
        getchar();
        return -1;
    }

    if(argc < 2)
    {
        printf("Error! Usage: ./app train for training or ./app test or predicting\n");
        return 0;
    }

    if(!strcmp(argv[1], "train")) {
        training_about();

        report.open(training_report_fn.c_str(), ios::out);
        image.open(training_image_fn.c_str(), ios::in | ios::binary); // Binary image file
        label.open(training_label_fn.c_str(), ios::in | ios::binary ); // Binary label file

        // reading file headers
        char number;
        for (int i = 1; i <= 16; ++i) {
            image.read(&number, sizeof(char));
        }
        for (int i = 1; i <= 8; ++i) {
            label.read(&number, sizeof(char));
        }

        init_array(global_eid);

        for (int sample = 1; sample <= nTraining; ++sample) {
            cout << "Sample " << sample << endl;

            // Getting (image, label)
            int label;
            char data[1024];
            ocall_read_one(data);
            input(global_eid, &label, data);
            // input_test();

            // Learning process: Perceptron (Forward procedure) - Back propagation
            int nIterations;
            learning_process(global_eid, &nIterations);

            // Write down the squared error
            cout << "No. iterations: " << nIterations << endl;
            double error;
            square_error(global_eid, &error);
            printf("Error: %0.6lf\n\n", error);

        }

        save_nn(global_eid);

    } else if(!strcmp(argv[1], "test")) {
        testing_about();


    /*
        sgx_status_t ret;
        char *message = "0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000111111000000000000000000000011111111111111110000000000001111111111111111000000000000000001111111111100000000000000000000000011110000000000000000000000011110000000000000000000000001111000000000000000000000001111000000000000000000000000111100000000000000000000000111100000000000000000000000011100000000000000000000000011110000000000000000000000011110000000000000000000000011111000000000000000000000001111000000000000000000000001111100000000000000000000000111100000000000000000000000111110000000000000000000000011111000000000000000000000001111000000000000000000000000000000000000000000";
        printf("Original message: %s\n", message);
        size_t encMessageLen = (SGX_AESGCM_MAC_SIZE + SGX_AESGCM_IV_SIZE + strlen(message));
        char *encMessage = (char *) malloc((encMessageLen+1)*sizeof(char));
        printf("Encrypting...\n");
        ret = encryptMessage(global_eid, message, strlen(message), encMessage, encMessageLen);
        encMessage[encMessageLen] = '\0';
        printf("Encrypted message: %s\n", encMessage);
        size_t decMessageLen = strlen(message);
        char *decMessage = (char *) malloc((decMessageLen+1)*sizeof(char));
        printf("Decrypting...\n");
        ret = decryptMessage(global_eid,encMessage,encMessageLen,decMessage,decMessageLen);
        decMessage[decMessageLen] = '\0';
        printf("Decrypted message: %s", decMessage);
    */


        report.open(testing_report_fn.c_str(), ios::out);
        image.open(testing_image_fn.c_str(), ios::in | ios::binary); // Binary image file
        label.open(testing_label_fn.c_str(), ios::in | ios::binary ); // Binary label file

        // Reading file headers
        char number;
        for (int i = 1; i <= 16; ++i) {
            image.read(&number, sizeof(char));
        }
        for (int i = 1; i <= 8; ++i) {
            label.read(&number, sizeof(char));
        }

        load_model(global_eid);

        int nCorrect = 0;
        for (int sample = 1; sample <= nTesting; ++sample) {
            cout << "Sample " << sample << endl;

            char data[1024];
            ocall_read_one(data);
            //printf("Data read from file: %s\n", data);

            // Getting (image, label)
            int label;
            input(global_eid, &label, data);
            // cout << "label: " << label << endl;

            // Classification - Perceptron procedure
            int result;
            predict(global_eid, &result);

            // Write down the classification result and the squared error
            double error;
            square_error(global_eid, &error);
            printf("Error: %0.6lf\n", error);

            if (label == result) {
                ++nCorrect;
                cout << "Classification: YES. Label = " << label << ". Predict = " << result << endl << endl;
            } else {
                cout << "Classification: NO.  Label = " << label << ". Predict = " << result << endl;
            }
        }

        // Summary
        double accuracy = (double)(nCorrect) / nTesting * 100.0;
        cout << "Number of correct samples: " << nCorrect << " / " << nTesting << endl;
        printf("Accuracy: %0.2lf\n", accuracy);
    }
    else {
        printf("Usage: ./app train for training, ./app test for predicting\n");
    }

    // printf_helloworld(global_eid);

    /* Destroy the enclave */
    sgx_destroy_enclave(global_eid);

    report.close();
    image.close();
    label.close();

    return 0;
}

