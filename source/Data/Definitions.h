#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "CellTypeConstants.h"

struct SimulationParameters;
struct RadiationSource;
struct ExpertToggles;
struct ColorTransitionRule;
struct ParameterSpec;

struct ContentDesc;
struct ObjectDesc;
struct ExtendedObjectDesc;
struct EnergyDesc;
struct GenomeDesc;
struct GeneDesc;

class SpaceCalculator;

class StatisticsHistory;

struct PreviewDesc;

struct ParametersFilter;

#if defined(__CUDACC__) || defined(__HIPCC__)
#define HOST_DEVICE __host__ __device__ __inline__
#else
#define HOST_DEVICE inline
#endif
