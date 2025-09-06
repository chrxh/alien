#pragma once

#include <memory>

class _SimulationCudaFacade;
using CudaSimulationFacade = std::shared_ptr<_SimulationCudaFacade>;

class _CollectionTOProvider;
using CollectionTOProvider = std::shared_ptr<_CollectionTOProvider>;
