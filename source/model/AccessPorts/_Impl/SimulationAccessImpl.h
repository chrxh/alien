#ifndef SIMULATIONACCESSIMPL_H
#define SIMULATIONACCESSIMPL_H

#include "model/AccessPorts/SimulationAccess.h"
#include "model/context/SimulationContext.h"
#include "SimulationAccessSlotWrapper.h"

template<typename DataDescriptionType>
class SimulationAccessImpl
	: public SimulationAccess<DataDescriptionType>
	, public SimulationAccessSlotWrapper
{
public:
	SimulationAccessImpl(QObject* parent = nullptr) : SimulationAccessSlotWrapper(parent) {}
	virtual ~SimulationAccessImpl() = default;

	virtual void init(SimulationContextApi* context) override;

	virtual void addData(DataDescriptionType const &desc) override;
	virtual void removeData(DataDescriptionType const &desc) override;
	virtual void updateData(DataDescriptionType const &desc) override;

	virtual void requestData(IntRect rect) override;
	virtual DataDescriptionType const& retrieveData() override;

private:
	virtual void accessToSimulation() override;

	SimulationContext* _context = nullptr;
	DataDescriptionType _dataToRetrieve;
	DataDescriptionType _dataToAdd;
};

#endif // SIMULATIONACCESSIMPL_H
