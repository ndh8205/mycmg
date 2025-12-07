# Hexarotor Simulation Makefile

CXX = g++
NVCC = nvcc
CXXFLAGS = -std=c++14 -O3 -Wall -Wextra -I./include -fopenmp
NVCCFLAGS = -std=c++14 -O3 -I./include -arch=sm_72 -Xcompiler -fopenmp
LDFLAGS = -lm -fopenmp
CUDA_LDFLAGS = -L/usr/local/cuda/lib64 -lcudart -lcurand

SRC_DIR = src
OBJ_DIR = obj
BIN_DIR = .

# Source files
COMMON_SRCS = $(SRC_DIR)/control_allocator.cpp \
              $(SRC_DIR)/controllers.cpp \
              $(SRC_DIR)/dynamics.cpp \
              $(SRC_DIR)/integrator.cpp \
              $(SRC_DIR)/disturbance.cpp

MAIN_ATT_PID_SRCS = $(COMMON_SRCS) $(SRC_DIR)/main_att_pid.cpp
MAIN_POS_PID_SRCS = $(COMMON_SRCS) $(SRC_DIR)/main_pos_pid.cpp
MAIN_ATT_SMC_SRCS = $(COMMON_SRCS) $(SRC_DIR)/main_att_smc.cpp
MAIN_POS_SMC_SRCS = $(COMMON_SRCS) $(SRC_DIR)/main_pos_smc.cpp
MAIN_ATT_SMC_DIST_SRCS = $(COMMON_SRCS) $(SRC_DIR)/main_att_smc_dist.cpp
TEST_DIST_SRCS = $(SRC_DIR)/disturbance.cpp $(SRC_DIR)/test_disturbance.cpp
MAIN_MPPI_HOVER_SRCS = $(COMMON_SRCS) $(SRC_DIR)/mppi_controller.cpp $(SRC_DIR)/main_mppi_hover.cpp
MAIN_MPPI_ATTITUDE_SRCS = $(COMMON_SRCS) $(SRC_DIR)/mppi_controller.cpp $(SRC_DIR)/main_mppi_attitude.cpp
MAIN_MPPI_CUDA_SRCS = $(COMMON_SRCS) $(SRC_DIR)/main_mppi_hover_cuda.cpp
MAIN_MPPI_ATTITUDE_CUDA_SRCS = $(COMMON_SRCS) $(SRC_DIR)/main_mppi_attitude_cuda.cpp
MPPI_CUDA_CU = $(SRC_DIR)/mppi_cuda.cu

# Object files
MAIN_ATT_PID_OBJS = $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(MAIN_ATT_PID_SRCS))
MAIN_POS_PID_OBJS = $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(MAIN_POS_PID_SRCS))
MAIN_ATT_SMC_OBJS = $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(MAIN_ATT_SMC_SRCS))
MAIN_POS_SMC_OBJS = $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(MAIN_POS_SMC_SRCS))
MAIN_ATT_SMC_DIST_OBJS = $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(MAIN_ATT_SMC_DIST_SRCS))
MAIN_MPPI_HOVER_OBJS = $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(MAIN_MPPI_HOVER_SRCS))
MAIN_MPPI_ATTITUDE_OBJS = $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(MAIN_MPPI_ATTITUDE_SRCS))
MAIN_MPPI_CUDA_OBJS = $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(MAIN_MPPI_CUDA_SRCS))
MAIN_MPPI_ATTITUDE_CUDA_OBJS = $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(MAIN_MPPI_ATTITUDE_CUDA_SRCS))
MPPI_CUDA_OBJ = $(OBJ_DIR)/mppi_cuda.o

# Targets
TARGET_ATT_PID = hexa_att_pid
TARGET_POS_PID = hexa_pos_pid
TARGET_ATT_SMC = hexa_att_smc
TARGET_POS_SMC = hexa_pos_smc
TARGET_ATT_SMC_DIST = hexa_att_smc_dist
TARGET_MPPI_HOVER = hexa_mppi_hover
TARGET_MPPI_ATTITUDE = hexa_mppi_attitude
TARGET_MPPI_CUDA = hexa_mppi_cuda
TARGET_MPPI_ATTITUDE_CUDA = hexa_mppi_attitude_cuda

.PHONY: all clean attitude_pid position_pid attitude_smc position_smc attitude_smc_dist mppi_hover mppi_attitude mppi_cuda mppi_attitude_cuda

all: attitude_pid position_pid attitude_smc position_smc attitude_smc_dist mppi_hover mppi_attitude

cuda: mppi_cuda mppi_attitude_cuda

attitude_pid: $(BIN_DIR)/$(TARGET_ATT_PID)

position_pid: $(BIN_DIR)/$(TARGET_POS_PID)

attitude_smc: $(BIN_DIR)/$(TARGET_ATT_SMC)

position_smc: $(BIN_DIR)/$(TARGET_POS_SMC)

attitude_smc_dist: $(BIN_DIR)/$(TARGET_ATT_SMC_DIST)

mppi_hover: $(BIN_DIR)/$(TARGET_MPPI_HOVER)

mppi_attitude: $(BIN_DIR)/$(TARGET_MPPI_ATTITUDE)

mppi_cuda: $(BIN_DIR)/$(TARGET_MPPI_CUDA)

mppi_attitude_cuda: $(BIN_DIR)/$(TARGET_MPPI_ATTITUDE_CUDA)

# Build rules
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# CUDA build rule
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu | $(OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(BIN_DIR)/$(TARGET_ATT_PID): $(MAIN_ATT_PID_OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

$(BIN_DIR)/$(TARGET_POS_PID): $(MAIN_POS_PID_OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

$(BIN_DIR)/$(TARGET_ATT_SMC): $(MAIN_ATT_SMC_OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

$(BIN_DIR)/$(TARGET_POS_SMC): $(MAIN_POS_SMC_OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

$(BIN_DIR)/$(TARGET_ATT_SMC_DIST): $(MAIN_ATT_SMC_DIST_OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

$(BIN_DIR)/$(TARGET_MPPI_HOVER): $(MAIN_MPPI_HOVER_OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

$(BIN_DIR)/$(TARGET_MPPI_ATTITUDE): $(MAIN_MPPI_ATTITUDE_OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

$(BIN_DIR)/$(TARGET_MPPI_CUDA): $(MAIN_MPPI_CUDA_OBJS) $(MPPI_CUDA_OBJ)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) $(CUDA_LDFLAGS)

$(BIN_DIR)/$(TARGET_MPPI_ATTITUDE_CUDA): $(MAIN_MPPI_ATTITUDE_CUDA_OBJS) $(MPPI_CUDA_OBJ)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) $(CUDA_LDFLAGS)

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)/$(TARGET_ATT_PID) $(BIN_DIR)/$(TARGET_POS_PID) $(BIN_DIR)/$(TARGET_ATT_SMC) $(BIN_DIR)/$(TARGET_POS_SMC) $(BIN_DIR)/$(TARGET_ATT_SMC_DIST) $(BIN_DIR)/$(TARGET_MPPI_HOVER) $(BIN_DIR)/$(TARGET_MPPI_ATTITUDE) $(BIN_DIR)/$(TARGET_MPPI_CUDA) $(BIN_DIR)/$(TARGET_MPPI_ATTITUDE_CUDA)
	rm -f *.csv *.png

.PHONY: run
run: $(BIN_DIR)/$(TARGET_ATT_PID)
	./$(TARGET_ATT_PID)
	python3 python/visualize_hexa.py
