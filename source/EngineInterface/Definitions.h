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

struct Description;
struct CellDescription;
struct ParticleDescription;
struct GenomeDescription;
struct GeneDescription;

struct CudaSettings;

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

struct PreviewDescription;

struct ConversionResult;

struct Ids;

class _OpenGlGeometryBuffers;
using OpenGlGeometryBuffers = std::shared_ptr<_OpenGlGeometryBuffers>;

struct NumRenderObjects;
struct CellVertexData;
struct EnergyParticleVertexData;

struct ParametersFilter;