#ifndef SEALING_H_
#define SEALING_H_

#include "sgx_trts.h"
#include "sgx_tseal.h"

#include "../../App/App.h"

sgx_status_t seal_nn(const NeuralNetwork* plaintext, sgx_sealed_data_t* sealed_data, size_t sealed_size);

sgx_status_t seal_test(const int* plaintext, sgx_sealed_data_t* sealed_data, size_t sealed_size);

sgx_status_t unseal_nn(const sgx_sealed_data_t* sealed_data, NeuralNetwork* plaintext, uint32_t plaintext_size);


#endif // SEALING_H_


