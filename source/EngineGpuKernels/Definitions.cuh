#pragma once

#include <memory>

#include <EngineInterface/CellTypeConstants.h>

#include <cuda/helper_cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

struct Cell;
struct Token;
struct Particle;
struct Objects;
struct Creature;
struct Genome;

struct SimulationData;
struct CudaGeometryBuffers;
class SelectionResult;
struct CellTO;
struct ClusterAccessTO;
struct TO;
struct SimulationParameters;
struct CudaSettings;
class SimulationStatistics;

class _SimulationKernelsService;
using SimulationKernelsService = std::shared_ptr<_SimulationKernelsService>;

class _DataAccessKernelsService;
using DataAccessKernelsService = std::shared_ptr<_DataAccessKernelsService>;

class _GarbageCollectorKernelsService;
using GarbageCollectorKernelsService = std::shared_ptr<_GarbageCollectorKernelsService>;

class _GeometryKernelsService;
using GeometryKernelsService = std::shared_ptr<_GeometryKernelsService>;

class _EditKernelsService;
using EditKernelsService = std::shared_ptr<_EditKernelsService>;

class _SelectionKernelsService;
using SelectionKernelsService = std::shared_ptr<_SelectionKernelsService>;

class _StatisticsKernelsService;
using StatisticsKernelsService = std::shared_ptr<_StatisticsKernelsService>;

class _TestKernelsService;
using TestKernelsService = std::shared_ptr<_TestKernelsService>;

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
