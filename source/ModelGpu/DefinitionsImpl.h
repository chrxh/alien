#pragma once

class SimulationControllerGpuImpl;
class SimulationContextGpuImpl;
class WorkerForGpu;
class GpuObserver;
class ThreadController;
	
enum RunningMode { DoNothing, CalcSingleTimestep, OpenEndedSimulation };
