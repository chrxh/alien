#pragma once

#include <QThread>

#include "ModelBasic/Definitions.h"
#include "DefinitionsImpl.h"

class CudaController
	: public QObject
{
	Q_OBJECT
public:
	CudaController(QObject* parent = nullptr);
	virtual ~CudaController();

	void init(SpaceProperties *metric);

	CudaWorker* getCudaWorker() const;

	void calculate(RunningMode mode);
	void restrictTimestepsPerSecond(optional<int> tps);

	Q_SIGNAL void timestepCalculated();

private:
	Q_SIGNAL void runWorker();
	Q_SLOT void timestepCalculatedWithGpu();

	SpaceProperties *_metric = nullptr;

	QThread _thread;
	CudaWorker* _worker = nullptr;
	bool _gpuThreadWorking = false;
};
