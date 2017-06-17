#pragma once

class SimulationControllerGpuImpl;
class SimulationContextGpuImpl;
class GpuWorker;
class GpuObserver;
class GpuThreadController;
	
enum RunningMode { DoNothing, CalcSingleTimestep, OpenEndedSimulation };
