#ifndef SIMULATIONACCESSIMPL_H
#define SIMULATIONACCESSIMPL_H

#include "model/tools/SimulationAccess.h"

class SimulationAccessImpl
	: public SimulationAccess
{
	Q_OBJECT
public:
	SimulationAccessImpl(QObject* parent = nullptr) : SimulationAccess(parent) {}
	virtual ~SimulationAccessImpl() = default;

	virtual void init(SimulationContextApi* context) override;

	virtual void addCell(CellDescription desc) override;

	virtual std::vector<UnitContextApi*> getAndLockData(IntRect rect) override;
	virtual void unlock() override;

private:
	SimulationContext* _context = nullptr;
};

#endif // SIMULATIONACCESSIMPL_H
