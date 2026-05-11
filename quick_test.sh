#!/bin/bash

cmake --build build --target clean

cmake -S . -B build -DBUILD_TESTS=ON -DENABLE_AVX512=ON -DENABLE_ASAN=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON && \
    cmake --build build -j4 &&\
    cd build &&\
    ctest --output-on-failure --verbose &&\
    cd ..
