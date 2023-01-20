all: conv_100

CUPTI_PATH=/usr/local/cuda/extras/CUPTI
INCLUDES = -I ../include -I /usr/local/cuda/extras/CUPTI/include
CXXARGS = -std=c++11 -g
CXXARGS += -Xcompiler -DNDEBUG
LIBS = -lcuda -L$(CUPTI_PATH)/lib64 -lcupti


conv : CUPTI_Conv100.cu
	nvcc $(CXXARGS) $(INCLUDES) $(LIBS) CUPTI_Conv100.cu -o conv

clean: 
	rm -f *.o conv conv_100
