#pragma once

#include <cstdint>
#include <map>
#include <string>

#include "Base/Definitions.h"
#include "Enums.h"

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

struct OverallStatistics;
class SpaceCalculator;