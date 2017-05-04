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

	virtual void init(SimulationContextHandle* context) override;
	Q_SLOT virtual void setRun(bool run) override;

	virtual MapManipulator* getMapManipulator() const override;

private:
	SimulationContext* _context = nullptr;
	MapManipulator* _manipulator = nullptr;
};

#endif // SIMULATIONCONTROLLERIMPL_H
