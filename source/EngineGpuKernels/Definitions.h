#pragma once

#include <memory>

class _SimulationCudaFacade;
using CudaSimulationFacade = std::shared_ptr<_SimulationCudaFacade>;

class _TOProvider;
using TOProvider = std::shared_ptr<_TOProvider>;
