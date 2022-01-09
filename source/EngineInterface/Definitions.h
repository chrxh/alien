#pragma once

#include <cstdint>
#include <map>
#include <string>

#include "Base/Definitions.h"
#include "ElementaryTypes.h"

struct SimulationParameters;

struct DataDescription;
struct ClusterDescription;
struct CellDescription;
struct ParticleDescription;

struct GpuSettings;

struct GeneralSettings;
struct Settings;

class _Serializer;
using Serializer = std::shared_ptr<_Serializer>;

class _SimulationController;
using SimulationController = std::shared_ptr<_SimulationController>;

struct OverallStatistics;
