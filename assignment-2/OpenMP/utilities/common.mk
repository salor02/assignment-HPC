ifndef CUDA_HOME
CUDA_HOME:=/usr/local/cuda
endif
NVCC=$(CUDA_HOME)/bin/nvcc
NVOPT:=-Xcompiler -fopenmp -lineinfo 
LDFLAGS:=-lm -lcudart $(EXT_LDFLAGS)
NVCFLAGS:=$(CXXFLAGS) $(NVOPT)
NVLDFLAGS:=$(LDFLAGS) -lgomp


INCPATHS = -I$(UTIL_DIR)

BENCHMARK = $(shell basename `pwd`)
EXE = $(BENCHMARK)_acc
SRC = $(BENCHMARK).c
HEADERS = $(BENCHMARK).h

SRC += $(UTIL_DIR)/polybench.c

DEPS        := Makefile.dep
DEP_FLAG    := -MM

CC=clang
LD=ld
OBJDUMP=objdump

OPT=-O0 -g
OMP=-fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda
CFLAGS=$(OPT) $(OMP) -I. $(EXT_CFLAGS)
# LDFLAGS=-lm $(EXT_LDFLAGS) VIENE MESSO GIà SOPRA

$(EXE):	$(OBJS)
	$(MKDIR_P) $(dir $@)
	$(NVCC) $(NVCFLAGS) $(OBJS) -o $@ $(NVLDFLAGS)

$(BUILD_DIR)/%.cu.o: %.cu
	$(MKDIR_P) $(dir $@)
	$(NVCC) $(NVCFLAGS) -c $< -o $@

$(BUILD_DIR)/%.cpp.o: %.cpp
	$(MKDIR_P) $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/%.c.o: %.c
	$(MKDIR_P) $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@


.PHONY: all exe clean veryclean

all : exe

.PHONY: run profile clean
run: $(EXE)
	./$(EXE)

profile: $(EXE)
	sudo $(CUDA_HOME)/bin/nvprof --unified-memory-profiling off ./$(EXE)

metrics: $(EXE)
	sudo $(CUDA_HOME)/bin/nvprof --print-gpu-trace --metrics "eligible_warps_per_cycle,achieved_occupancy,sm_efficiency,ipc" ./$(EXE)

clean:
	-rm -fr $(BUILD_DIR) *.exe *.out *~

MKDIR_P ?= mkdir -p

$(DEPS): $(SRC) $(HEADERS)
	$(CC) $(INCPATHS) $(DEP_FLAG) $(SRC) > $(DEPS)

-include $(DEPS)
