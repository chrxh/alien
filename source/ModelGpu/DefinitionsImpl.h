#pragma once

class SimulationControllerGpuImpl;
class SimulationContextGpuImpl;
class CudaWorker;
class GpuObserver;
class ThreadController;
	
enum RunningMode {
	DoNothing, 
	CalcSingleTimestep, 
	OpenEndedSimulation
};
