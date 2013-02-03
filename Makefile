#########################################################
# Audio Makefile
# 
# Makefile used for compiling and linking source code
# associated with realtime audio (using portaudio 
# library)
#
# Adam Miller
# 9/14/12
#########################################################

# Setup search paths
LIB_DIR := ./lib/
BIN_DIR := ./bin/
DSP_LIB_DIR := /System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Versions/A
vpath %.c ./src/examples ./src ./portaudio/src/common \
			./portaudio/src/os/unix/
vpath %.h ./include
vpath %.o ./lib

# Setup compiler and linker
CC := gcc
C_FLAGS := -c -I ./portaudio/include -I ./portaudio/src/common \
			-I ./portaudio/src/unix/ -I ./portaudio/src/mac_osx/ \
			-I/opt/local/include -I./include/ -std=c99 -g \
			-I/System/Library/Frameworks/vecLib.framework/Versions/A/Headers
LINKER := gcc
L_FLAGS := -L/opt/local/lib -B$(LIB_DIR) -lportaudio -L./portaudio/src/common/ 

# List of different targets
TARGETS := sin record exrecord realtime realtime_sin realtime_dft \
			realtime_stft plot

# Setup required objs for different targets
SIN_OBJS :=  sin.o audio_tools.o # for target: sin
REC_OBJS :=  record.o audio_tools.o # for target: record
REALTIME_OBJS := realtime.o audio_tools.o # for target: realtime
REALTIME_BUF_OBJS := realtime_buf.o audio_tools.o pa_ringbuffer.o 
REALTIME_DFT_OBJS := realtime_dft.o audio_tools.o pa_ringbuffer.o \
					 realtimestft.o
PLOT_OBJS := plot.o

# Specify targets/recipes
.PHONY: all
all: $(TARGETS)

sin: $(SIN_OBJS)
	$(LINKER) $(L_FLAGS)  $(addprefix $(LIB_DIR), $(notdir $^)) -o $@

record: $(REC_OBJS)
	$(LINKER) $(L_FLAGS) $(addprefix $(LIB_DIR), $(notdir $^)) -o $@

exrecord: paex_record.o
	$(LINKER) $(L_FLAGS) $(addprefix $(LIB_DIR), $(notdir $^)) -o $@

realtime: $(REALTIME_OBJS)
	$(LINKER) $(L_FLAGS) $(addprefix $(LIB_DIR), $(notdir $^)) -o $@

realtime_buf: $(REALTIME_BUF_OBJS)
	$(LINKER) $(L_FLAGS) $(addprefix $(LIB_DIR), $(notdir $^)) -o $@

realtime_dft: $(REALTIME_DFT_OBJS)
	$(LINKER) $(L_FLAGS) -L $(DSP_LIB_DIR) -lvDSP \
	$(addprefix $(LIB_DIR), $(notdir $^)) -o $@

plot: $(PLOT_OBJS)
	$(LINKER) $(L_FLAGS) -lplplotd $(addprefix $(LIB_DIR), $(notdir $^)) -o $@

%.o: %.c
	$(CC) $(C_FLAGS) $^ -o $(addprefix $(LIB_DIR), $@)

.PHONY: clean
clean:
	rm -f $(TARGETS)
	rm -f $(addprefix $(BIN_DIR), $(TARGETS))
	rm -f $(LIB_DIR)*.o
	rm -f *.o
	rm -f output?.dat
