#pragma once

class SimulationControllerGpuImpl;
class SimulationContextGpuImpl;
class GpuWorker;
class GpuObserver;
class ThreadController;
	
enum RunningMode {
	DoNothing, 
	CalcSingleTimestep, 
	OpenEndedSimulation
};
