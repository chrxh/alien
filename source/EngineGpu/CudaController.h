#pragma once

#include <QThread>

#include "EngineInterface/Definitions.h"
#include "DefinitionsImpl.h"

class CudaController
	: public QObject
{
	Q_OBJECT
public:
	CudaController(QObject* parent = nullptr);
	virtual ~CudaController();

    void init(
        SpaceProperties* space,
        int timestep,
        SimulationParameters const& parameters,
        CudaConstants const& cudaConstants);

    CudaWorker* getCudaWorker() const;

	void calculate(RunningMode mode);
	void restrictTimestepsPerSecond(optional<int> tps);
	void setSimulationParameters(SimulationParameters const& parameters);
    void setExecutionParameters(ExecutionParameters const& parameters);

	Q_SIGNAL void timestepCalculated();

private:
	Q_SIGNAL void runWorker();
    Q_SLOT void errorThrown(QString message);
	Q_SLOT void timestepCalculatedWithGpu();

	SpaceProperties *_metric = nullptr;

	QThread _thread;
	CudaWorker* _worker = nullptr;
    NumberGenerator* _numberGenerator = nullptr;
	bool _gpuThreadWorking = false;
};
