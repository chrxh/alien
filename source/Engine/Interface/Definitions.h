#pragma once

#include <array>
#include <cstdint>
#include <memory>

#include <Data/Definitions.h>

struct KernelLaunchSettings;

struct SettingsForSimulation;

class _SimulationFacade;
using SimulationFacade = std::shared_ptr<_SimulationFacade>;

struct ConversionResult;

class _GeometryBuffers;
using GeometryBuffers = std::shared_ptr<_GeometryBuffers>;

// Identifies the GPU across graphics and compute APIs
using GpuUuid = std::array<uint8_t, 16>;

struct NumRenderObjects;
struct ObjectVertexData;
struct FluidParticleVertexData;
