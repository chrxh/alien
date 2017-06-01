#ifndef SIMULATIONCONTROLLERIMPL_H
#define SIMULATIONCONTROLLERIMPL_H

#include <QTime>
#include "Model/SimulationController.h"

class SimulationControllerImpl
	: public SimulationController
{
	Q_OBJECT
public:
	SimulationControllerImpl(QObject* parent = nullptr);
	virtual ~SimulationControllerImpl() = default;

	virtual void init(SimulationContextApi* context) override;
	virtual void setRun(bool run) override;
	virtual void calculateSingleTimestep() override;
	virtual SimulationContextApi* getContext() const override;

private:
	Q_SLOT void oneSecondTimerTimeout();

	SimulationContext* _context = nullptr;

	bool _flagSimulationRunning = false;
	QTimer* _oneSecondTimer = nullptr;
	int _timestepsPerSecond = 0;
	QTime _timeSinceLastStart;
	int _displayedFramesSinceLastStart = 0;
};

#endif // SIMULATIONCONTROLLERIMPL_H
