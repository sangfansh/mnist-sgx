# mnist-sgx

The docker file I used is from this repo: https://github.com/tozd/docker-sgx

First, install the sgx driver: https://github.com/intel/linux-sgx-driver

Then run the docker file:
docker run -d --device /dev/isgx --device /dev/mei0 --name test-sgx tozd/sgx:ubuntu-xenial
docker exec -t -i test-sgx bash

Maybe also need to run this command inside docker:
source /opt/intel/sgxsdk/environment

Then clone this repo into the docker and ru the app:
./app train to train the neural network
./app test to test some prediction
