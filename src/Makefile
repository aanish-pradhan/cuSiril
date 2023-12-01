STACKING_HEADER_FILES = -I ../lib/include/Sorting/ -I ../lib/include/Stacking/ -I ../lib/include/Statistics/
STACKING_IMPLEMENTATION_FILES = Stacking/stack.cu Stacking/stack_kernels.cu Statistics/statistics.cu Sorting/quickselect.cu

make stacking:
	nvcc -arch=sm_75 $(STACKING_HEADER_FILES) -O3 -o Stacking/stack.so --relocatable-device-code=true -shared -Xcompiler -fPIC $(STACKING_IMPLEMENTATION_FILES)