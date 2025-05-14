#pragma once

#include <memory>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda/helper_cuda.h>

#include "EngineInterface/CellTypeConstants.h"

struct Cell;
struct Token;
struct Particle;
struct Objects;

struct SimulationData;
struct RenderingData;
class SelectionResult;
struct CellTO;
struct ClusterAccessTO;
struct DataTO;
struct SimulationParameters;
struct GpuSettings;
class SimulationStatistics;

class _SimulationKernelsService;
using SimulationKernelsService = std::shared_ptr<_SimulationKernelsService>;

class _DataAccessKernelsService;
using DataAccessKernelsService = std::shared_ptr<_DataAccessKernelsService>;

class _GarbageCollectorKernelsService;
using GarbageCollectorKernelsService = std::shared_ptr<_GarbageCollectorKernelsService>;

class _RenderingKernelsService;
using RenderingKernelsService = std::shared_ptr<_RenderingKernelsService>;

class _EditKernelsService;
using EditKernelsService = std::shared_ptr<_EditKernelsService>;

class _StatisticsKernelsService;
using StatisticsKernelsService = std::shared_ptr<_StatisticsKernelsService>;

class _TestKernelsService;
using TestKernelsService = std::shared_ptr<_TestKernelsService>;

class _MaxAgeBalancer;
using MaxAgeBalancer = std::shared_ptr<_MaxAgeBalancer>;

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
