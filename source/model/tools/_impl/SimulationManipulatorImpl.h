#ifndef MAPMANIPULATORIMPL_H
#define MAPMANIPULATORIMPL_H

#include "model/tools/SimulationManipulator.h"

class SimulationManipulatorImpl
	: public SimulationManipulator
{
	Q_OBJECT
public:
	SimulationManipulatorImpl(QObject* parent = nullptr) : SimulationManipulator(parent) {}
	virtual ~SimulationManipulatorImpl() = default;

	virtual void init(SimulationContextApi* context) override;

	virtual void addCell(CellDescription desc) override;

private:
	SimulationContext* _context = nullptr;
};

#endif // MAPMANIPULATORIMPL_H
