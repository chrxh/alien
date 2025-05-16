#pragma once

#include <memory>

class _SimulationCudaFacade;
using CudaSimulationFacade = std::shared_ptr<_SimulationCudaFacade>;

class _DataTOCache;
using DataTOCache = std::shared_ptr<_DataTOCache>;
