#pragma once

class SimulationControllerGpuImpl;
class SimulationContextGpuImpl;
class CudaWorker;
class GpuObserver;
class CudaController;
struct CudaConstants;
class ModelGpuData;

class _CudaJob;
using CudaJob = boost::shared_ptr<_CudaJob>;

class _GetDataJob;
using GetDataJob = boost::shared_ptr<_GetDataJob>;

class _SetDataJob;
using SetDataJob = boost::shared_ptr<_SetDataJob>;

class _RunSimulationJob;
using RunSimulationJob = boost::shared_ptr<_RunSimulationJob>;

class _StopSimulationJob;
using StopSimulationJob = boost::shared_ptr<_StopSimulationJob>;

class _CalcSingleTimestepJob;
using CalcSingleTimestepJob = boost::shared_ptr<_CalcSingleTimestepJob>;

enum RunningMode {
	DoNothing, 
	CalcSingleTimestep, 
	OpenEndedSimulation
};
