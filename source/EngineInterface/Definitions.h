#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <memory>
#include <vector>

#include "CellTypeConstants.h"

struct SimulationParameters;
struct SimulationParametersZoneValues;
struct ColorTransitionRules;

struct ClusteredDataDescription;
struct DataDescription;
struct ClusterDescription;
struct CellDescription;
struct ParticleDescription;

struct GpuSettings;

struct SettingsForSimulation;

class _SimulationFacade;
using SimulationFacade = std::shared_ptr<_SimulationFacade>;

struct TimelineStatistics;
struct HistogramData;
struct StatisticsRawData;

class SpaceCalculator;

class _ShapeGenerator;
using ShapeGenerator = std::shared_ptr<_ShapeGenerator>;

class ShapeGeneratorResult;

class StatisticsHistory;

class SimulationParametersService;
