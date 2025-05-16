#pragma once

#include <memory>

class _SimulationCudaFacade;
using CudaSimulationFacade = std::shared_ptr<_SimulationCudaFacade>;

class _DataTOProvider;
using DataTOProvider = std::shared_ptr<_DataTOProvider>;
