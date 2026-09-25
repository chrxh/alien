#pragma once

#include <memory>

#include <Data/Definitions.h>

struct KernelLaunchSettings;

struct SettingsForSimulation;

class _SimulationFacade;
using SimulationFacade = std::shared_ptr<_SimulationFacade>;

class ShapeGenerator;

struct ShapeGeneratorResult;

struct ConversionResult;

class _GeometryBuffers;
using GeometryBuffers = std::shared_ptr<_GeometryBuffers>;

struct NumRenderObjects;
struct ObjectVertexData;
struct FluidParticleVertexData;
