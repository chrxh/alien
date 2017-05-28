#ifndef SIMULATIONACCESSIMPL_H
#define SIMULATIONACCESSIMPL_H

#include "Model/AccessPorts/SimulationAccess.h"
#include "Model/Context/SimulationContext.h"
#include "Model/Context/UnitObserver.h"
#include "Model/Entities/Descriptions.h"

class SimulationAccessImpl
	: public SimulationAccess
	, public UnitObserver
{
public:
	SimulationAccessImpl(QObject* parent = nullptr) : SimulationAccess(parent) {}
	virtual ~SimulationAccessImpl();

	virtual void init(SimulationContextApi* context) override;

	virtual void updateData(DataDescription const &desc) override;
	virtual void requireData(IntRect rect, ResolveDescription const& resolveDesc) override;
	virtual DataDescription const& retrieveData() override;

	virtual void unregister() override;
	virtual void accessToUnits() override;
private:
	void callBackUpdateData();
	void callBackCollectData();

	void collectDataFromUnit(Unit* unit);
	void collectClustersFromUnit(Unit* unit);
	void collectParticlesFromUnit(Unit* unit);

	SimulationContext* _context = nullptr;
	bool _registered = false;

	bool _dataRequired = false;
	IntRect _requiredRect;
	ResolveDescription _resolveDesc;
	DataDescription _dataCollected;
	DataDescription _dataToUpdate;
};

#endif // SIMULATIONACCESSIMPL_H
