#include "Enclave_t.h"
#include "sgx_trts.h"
#include "sgx_tseal.h"

#include "sealing.h"

sgx_status_t seal_nn(const NeuralNetwork* plaintext, sgx_sealed_data_t* sealed_data, size_t sealed_size) {
    return sgx_seal_data(0, NULL, sizeof(NeuralNetwork), (uint8_t*)plaintext, sealed_size, sealed_data);
}

sgx_status_t seal_test(const double* plaintext, sgx_sealed_data_t* sealed_data, size_t sealed_size) {
    return sgx_seal_data(0, NULL, 821472 , (uint8_t*)plaintext, sealed_size, sealed_data);
}

sgx_status_t unseal_nn(const sgx_sealed_data_t* sealed_data, NeuralNetwork* plaintext, uint32_t plaintext_size) {
    return sgx_unseal_data(sealed_data, NULL, NULL, (uint8_t*)plaintext, &plaintext_size);
}

