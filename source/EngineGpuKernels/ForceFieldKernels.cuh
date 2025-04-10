﻿#pragma once

#include "Math.cuh"
#include "Map.cuh"
#include "SimulationData.cuh"

__global__ void cudaApplyForceFieldSettings(SimulationData data);