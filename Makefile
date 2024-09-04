# Variables
CC = clang
CFLAGS = -std=c17
SRC_DIR = ./src
OUTPUT_DIR = ./output

# Find all .c files in src directory and subdirectories
SRC_FILES = $(shell find $(SRC_DIR) -name '*.c')

# Generate the corresponding .o files in the output directory
OBJS = $(patsubst $(SRC_DIR)/%, $(OUTPUT_DIR)/%, $(SRC_FILES:.c=.o))

# Collect all subdirectories under src for include paths
INCLUDES = $(shell find $(SRC_DIR) -type d -exec echo -I{} \;)

all: $(OUTPUT_DIR)/output.out

run: $(OUTPUT_DIR)/output.out
	clear && ./$(OUTPUT_DIR)/output.out

# Target for the final executable
$(OUTPUT_DIR)/output.out: $(OBJS)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $^

# Generic target for creating object files
$(OUTPUT_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(dir $@)  # Create directories as needed
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ $<

# Target for clean
clean:
	rm -rf $(OUTPUT_DIR)

# Target for .PHONY
.PHONY: all run clean
