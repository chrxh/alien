#pragma once

struct Cell;
struct Token;
struct Particle;
struct Entities;

struct SimulationData;
class SimulationResult;
struct CellAccessTO;
struct ClusterAccessTO;
struct DataAccessTO;
struct SimulationParameters;
struct GpuConstants;
class CudaMonitorData;

#define FP_PRECISION 0.00001

#define CUDA_THROW_NOT_IMPLEMENTED() printf("not implemented"); \
    asm("trap;");