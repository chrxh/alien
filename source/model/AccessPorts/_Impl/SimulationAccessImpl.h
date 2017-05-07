#ifndef SIMULATIONACCESSIMPL_H
#define SIMULATIONACCESSIMPL_H

#include "model/AccessPorts/SimulationAccess.h"

class SimulationAccessImpl
	: public SimulationAccess
{
	Q_OBJECT
public:
	SimulationAccessImpl(QObject* parent = nullptr) : SimulationAccess(parent) {}
	virtual ~SimulationAccessImpl() = default;

	virtual void init(SimulationContextApi* context) override;

	virtual void addCell(CellDescription desc) override;

private:
	SimulationContext* _context = nullptr;
};

#endif // SIMULATIONACCESSIMPL_H
