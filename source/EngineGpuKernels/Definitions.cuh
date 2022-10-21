#pragma once

#include <memory>

#include "EngineInterface/ArraySizes.h"

struct Cell;
struct Token;
struct Particle;
struct Objects;

struct SimulationData;
struct RenderingData;
class SimulationResult;
class SelectionResult;
struct CellTO;
struct ClusterAccessTO;
struct DataTO;
struct SimulationParameters;
struct GpuSettings;
class CudaMonitorData;

class _SimulationKernelsLauncher;
using SimulationKernelsLauncher = std::shared_ptr<_SimulationKernelsLauncher>;

class _DataAccessKernelsLauncher;
using DataAccessKernelsLauncher = std::shared_ptr<_DataAccessKernelsLauncher>;

class _GarbageCollectorKernelsLauncher;
using GarbageCollectorKernelsLauncher = std::shared_ptr<_GarbageCollectorKernelsLauncher>;

class _RenderingKernelsLauncher;
using RenderingKernelsLauncher = std::shared_ptr<_RenderingKernelsLauncher>;

class _EditKernelsLauncher;
using EditKernelsLauncher = std::shared_ptr<_EditKernelsLauncher>;

class _MonitorKernelsLauncher;
using MonitorKernelsLauncher = std::shared_ptr<_MonitorKernelsLauncher>;

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
