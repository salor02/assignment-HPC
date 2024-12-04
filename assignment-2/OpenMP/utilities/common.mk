# Prevent implicit rules
MAKEFLAGS += -r

CUDA_HOME := /usr/local/cuda
LIB_PATHS := /usr/ext/lib:$(CUDA_HOME)/lib

# Compiler and tools
CXX := gcc
NVCC := $(CUDA_HOME)/bin/nvcc

# Flags
DATASET_TYPE ?= MINI_DATASET        # Default dataset type
DUMP_ARRAYS ?= 0                    # Default: do not dump arrays (0)
OPTIMIZATION ?= SEQUENTIAL  		# Default: esecuzione sequenziale
CHECK_RESULTS ?= 0					# Default: non controllare il risultato


CXXFLAGS := -O0 -Xcompiler -fopenmp -DPOLYBENCH_TIME -DPOLYBENCH_USE_SCALAR_LB -D$(DATASET_TYPE) -D$(OPTIMIZATION)
NVCCFLAGS := $(CXXFLAGS) -lineinfo
LDFLAGS := -L$(CUDA_HOME)/lib -lcudart

# Add extra flags for dumping arrays
ifeq ($(DUMP_ARRAYS), 1)
CXXFLAGS += -DPOLYBENCH_DUMP_ARRAYS
NVCCFLAGS += -DPOLYBENCH_DUMP_ARRAYS
endif
ifeq ($(CHECK_RESULTS), 1)
CXXFLAGS += -DCHECK_RESULTS
NVCCFLAGS += -DCHECK_RESULTS
endif

# Enable debugging
DEBUG_FLAGS := -g -G
RELEASE_FLAGS := -O3

# Add extra flags via command line
CXXFLAGS += $(EXTRA_CXXFLAGS)
NVCCFLAGS += $(EXTRA_NVFLAGS)

# Source files and directories
SRCS := $(UTIL_DIR)/polybench.cu atax.cu
OBJDIR := obj
OBJS := $(addprefix $(OBJDIR)/, $(SRCS:.cu=.o))
EXE := atax_acc

# Targets
.PHONY: all clean profile run

# Default target
all: $(EXE)

run: $(EXE)
	./$(EXE)

# Linking
$(EXE): $(OBJS)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LDFLAGS)

# Compiling
$(OBJDIR)/%.o: %.cu | $(OBJDIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Ensure the object directory exists
$(OBJDIR):
	mkdir -p $@

# Profiling target
profile: $(EXE)
	LD_LIBRARY_PATH=$(LIB_PATHS):$(LD_LIBRARY_PATH) \
	LIBRARY_PATH=$(LIB_PATHS):$(LIBRARY_PATH) \
	sudo $(CUDA_HOME)/bin/nvprof ./$(EXE)

# Cleaning target
clean:
	rm -rf $(OBJDIR) $(EXE)
