#pragma once

#include <cstdint>
#include <map>
#include <string>

#include <boost/optional.hpp>
#include <boost/shared_ptr.hpp>

#include "ElementaryTypes.h"

struct SimulationParameters;

struct DataDescription2;
struct ClusterDescription2;
struct CellDescription2;
struct ParticleDescription2;

struct DataChangeDescription;
struct CellChangeDescription;
struct ParticleChangeDescription;

struct GpuSettings;

struct GeneralSettings;
struct Settings;

class _Serializer;
using Serializer = boost::shared_ptr<_Serializer>;

struct OverallStatistics;
