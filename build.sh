#!/bin/bash
nvcc -arch=sm_80 -O3 matrxMul.cu -o matrxMul
nvcc -arch=sm_80 -O3 mulVmma.cu -o mulVmma
nvcc -arch=sm_80 -O3 vmmaNative.cu -o vmmaNative