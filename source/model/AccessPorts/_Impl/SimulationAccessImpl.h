#ifndef SIMULATIONACCESSIMPL_H
#define SIMULATIONACCESSIMPL_H

#include "model/AccessPorts/SimulationAccess.h"
#include "model/context/SimulationContext.h"
#include "model/context/UnitObserver.h"

template<typename DataDescriptionType>
class SimulationAccessImpl
	: public SimulationAccess<DataDescriptionType>
	, public UnitObserver
{
public:
	SimulationAccessImpl(QObject* parent = nullptr) : SimulationAccess<DataDescriptionType>(parent) {}
	virtual ~SimulationAccessImpl();

	virtual void init(SimulationContextApi* context) override;

	virtual void addData(DataDescriptionType const &desc) override;
	virtual void removeData(DataDescriptionType const &desc) override;
	virtual void updateData(DataDescriptionType const &desc) override;

	virtual void requireData(IntRect rect) override;
	virtual DataDescriptionType const& retrieveData() override;

	virtual void unregister() override;
	virtual void accessToUnits() override;
private:
	void callBackAddData();
	void callBackGetData();

	SimulationContext* _context = nullptr;
	bool _registered = false;

	bool _dataRequired = false;
	IntRect _requiredRect;
	DataDescriptionType _dataCollected;
	DataDescriptionType _dataToAdd;
};

#endif // SIMULATIONACCESSIMPL_H
