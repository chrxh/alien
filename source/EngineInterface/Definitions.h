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

struct Desc;
struct ObjectDesc;
struct ExtendedObjectDesc;
struct EnergyDesc;
struct GenomeDesc;
struct GeneDesc;

struct CudaSettings;

struct SettingsForSimulation;

class _SimulationFacade;
using SimulationFacade = std::shared_ptr<_SimulationFacade>;

struct TimelineStatistics;
struct HistogramData;
struct StatisticsRawData;

class SpaceCalculator;

class ShapeGenerator;

struct ShapeGeneratorResult;

class StatisticsHistory;

struct PreviewDesc;

struct ConversionResult;

struct Ids;

class _GeometryBuffers;
using GeometryBuffers = std::shared_ptr<_GeometryBuffers>;

struct NumRenderObjects;
struct ObjectVertexData;
struct FluidParticleVertexData;

struct ParametersFilter;

#if defined(__CUDACC__)
#define HOST_DEVICE __host__ __device__
#else
#define HOST_DEVICE
#endif
