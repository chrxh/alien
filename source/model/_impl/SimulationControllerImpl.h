#ifndef SIMULATIONCONTROLLERIMPL_H
#define SIMULATIONCONTROLLERIMPL_H

#include "model/SimulationController.h"

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

	bool _flagSimulationRunning = false;
	SimulationContext* _context = nullptr;

	QTimer* _oneSecondTimer = nullptr;
	int _fps = 0;
};

#endif // SIMULATIONCONTROLLERIMPL_H
