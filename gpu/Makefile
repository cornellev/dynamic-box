OPT=-O1

CC := $(shell which nvcc)
INCLUDE_FLAGS := -I$(CONDA_PREFIX)/include/opencv4 -I$(CONDA_PREFIX)/include/eigen3
CONDA_LIB_DIR = $(CONDA_PREFIX)/lib
LD_FLAGS := -L$(CONDA_LIB_DIR) \
            -lopencv_core \
            -lopencv_highgui \
            -lopencv_imgproc \
            -lopencv_imgcodecs \
			-lopencv_stereo \
			-lopencv_calib3d
CPP_FLAGS := -std=c++17 --expt-relaxed-constexpr $(OPT)
EXEC_NAME := main

.PHONY: build
$(EXEC_NAME): main.cu
	$(CC) -o $(EXEC_NAME) main.cu $(CPP_FLAGS) $(INCLUDE_FLAGS) $(LD_FLAGS)

.PHONY: clean
clean:
	rm -f main
    
.PHONY: run
run: $(EXEC_NAME)
	LD_LIBRARY_PATH=$(CONDA_LIB_DIR):$(LD_LIBRARY_PATH) ./$(EXEC_NAME) display
