all:   p2p p2p_colocated

CUPTI_PATH=/usr/local/cuda/extras/CUPTI
INCLUDES = -I ../include -I /usr/local/cuda/extras/CUPTI/include -I ../Common

CXXARGS = -std=c++11 -g
CXXARGS += -Xcompiler -DNDEBUG
LIBS = -lcuda -L$(CUPTI_PATH)/lib64 -lcupti


conv_100 : CUPTI_Conv100.cu
	nvcc $(CXXARGS) $(INCLUDES) $(LIBS) CUPTI_Conv100.cu -o conv

nvlink : CUPTI_nvlink.cu
	nvcc $(CXXARGS) $(INCLUDES) $(LIBS) CUPTI_nvlink.cu -o nvlink

p2p : p2p.cu	
	nvcc $(CXXARGS) $(INCLUDES) $(LIBS) p2p.cu -o p2p

p2p_colocated : p2p_colocated.cu
	nvcc $(CXXARGS) $(INCLUDES) $(LIBS) p2p_colocated.cu -o p2p_colocated

clean: 
	rm -f *.o conv conv_100 nvlink p2p p2p_colocated
