#pragma once

#include <memory>

#include <EngineInterface/CellTypeConstants.h>

#include <cuda/helper_cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

struct Object;
struct Token;
struct Energy;
struct Entities;
struct Creature;
struct Genome;

struct SimulationData;
struct CudaGeometryBuffers;
class SelectionResult;
struct ObjectTO;
struct ClusterAccessTO;
struct TO;
struct SimulationParameters;
struct CudaSettings;
class SimulationStatistics;

class SimulationKernelsService;
class DataAccessKernelsService;
class GarbageCollectorKernelsService;
class GeometryKernelsService;
class EditKernelsService;
class SelectionKernelsService;
class StatisticsKernelsService;
class TestKernelsService;

class _MaxAgeBalancer;
using MaxAgeBalancer = std::shared_ptr<_MaxAgeBalancer>;

class _CudaTOProvider;
using CudaTOProvider = std::shared_ptr<_CudaTOProvider>;

class _TOProvider;
using TOProvider = std::shared_ptr<_TOProvider>;


struct ApplyForceData
{
    float2 startPos;
    float2 endPos;
    float2 force;
    float radius;
    bool onlyRotation;
};

struct PointSelectionData
{
    float2 pos;
    float radius;
};

struct AreaSelectionData
{
    float2 startPos;
    float2 endPos;
};
