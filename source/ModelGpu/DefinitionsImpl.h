#pragma once

class SimulationControllerGpuImpl;
class SimulationContextGpuImpl;
class CudaBridge;
class GpuObserver;
class ThreadController;
	
enum RunningMode {
	DoNothing, 
	CalcSingleTimestep, 
	OpenEndedSimulation
};
