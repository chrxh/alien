#pragma once

#include <cstdint>
#include <map>
#include <string>

#include "Base/Definitions.h"
#include "CellFunctionConstants.h"

struct SimulationParameters;

struct ClusteredDataDescription;
struct DataDescription;
struct ClusterDescription;
struct CellDescription;
struct ParticleDescription;

struct GpuSettings;

struct GeneralSettings;
struct Settings;

class _SimulationController;
using SimulationController = std::shared_ptr<_SimulationController>;

struct TimelineStatistics;
struct HistogramData;
struct StatisticsData;

class SpaceCalculator;

class _ShapeGenerator;
using ShapeGenerator = std::shared_ptr<_ShapeGenerator>;

class ShapeGeneratorResult;