#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <memory>
#include <vector>

#include "CellFunctionConstants.h"

struct SimulationParameters;

struct ClusteredDataDescription;
struct DataDescription;
struct ClusterDescription;
struct CellDescription;
struct ParticleDescription;

struct GpuSettings;

struct Settings;

class _SimulationFacade;
using SimulationFacade = std::shared_ptr<_SimulationFacade>;

struct TimelineStatistics;
struct HistogramData;
struct RawStatisticsData;

class SpaceCalculator;

class _ShapeGenerator;
using ShapeGenerator = std::shared_ptr<_ShapeGenerator>;

class ShapeGeneratorResult;

class StatisticsHistory;
