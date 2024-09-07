# Variables
CC = clang
NVCC = nvcc
CFLAGS = -std=gnu17
NVCCFLAGS = -std=c++11
SRC_DIR = ./src
OUTPUT_DIR = ./output

# Find all .c and .cu files in src directory and subdirectories
C_SRC_FILES = $(shell find $(SRC_DIR) -name '*.c')
CU_SRC_FILES = $(shell find $(SRC_DIR) -name '*.cu')

# Generate the corresponding .o files in the output directory
C_OBJS = $(patsubst $(SRC_DIR)/%, $(OUTPUT_DIR)/%, $(C_SRC_FILES:.c=.o))
CU_OBJS = $(patsubst $(SRC_DIR)/%, $(OUTPUT_DIR)/%, $(CU_SRC_FILES:.cu=.o))

# Collect all subdirectories under src for include paths
INCLUDES = $(shell find $(SRC_DIR) -type d -exec echo -I{} \;)

# All object files
OBJS = $(C_OBJS) $(CU_OBJS)

all: $(OUTPUT_DIR)/output.out

run: $(OUTPUT_DIR)/output.out
	clear && ./$(OUTPUT_DIR)/output.out

# Target for the final executable
$(OUTPUT_DIR)/output.out: $(OBJS)
	$(NVCC) $(OBJS) $(INCLUDES) -o $@

# Target for creating object files from .c files
$(OUTPUT_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(dir $@)  # Create directories as needed
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ $<

# Target for creating object files from .cu files
$(OUTPUT_DIR)/%.o: $(SRC_DIR)/%.cu
	@mkdir -p $(dir $@)  # Create directories as needed
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c -o $@ $<

# Target for clean
clean:
	rm -rf $(OUTPUT_DIR)

# Target for .PHONY
.PHONY: all run clean
