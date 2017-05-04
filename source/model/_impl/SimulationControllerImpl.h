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
	Q_SLOT virtual void setRun(bool run) override;

private:
	SimulationContext* _context = nullptr;
};

#endif // SIMULATIONCONTROLLERIMPL_H
