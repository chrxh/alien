#ifndef SIMULATIONACCESSIMPL_H
#define SIMULATIONACCESSIMPL_H

#include "model/AccessPorts/SimulationAccess.h"
#include "model/context/SimulationContext.h"

template<typename DataDescriptionType>
class SimulationAccessImpl
	: public SimulationAccess<DataDescriptionType>
{
public:
	virtual ~SimulationAccessImpl() = default;

	virtual void init(SimulationContextApi* context) override;

	virtual void addData(DataDescriptionType const &desc) override;
	virtual void removeData(DataDescriptionType const &desc) override;
	virtual void updateData(DataDescriptionType const &desc) override;
	virtual void getData(IntRect rect, DataDescriptionType& result) override;

private:
	SimulationContext* _context = nullptr;
};

#endif // SIMULATIONACCESSIMPL_H
