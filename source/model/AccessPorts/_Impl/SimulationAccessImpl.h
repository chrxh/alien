#ifndef SIMULATIONACCESSIMPL_H
#define SIMULATIONACCESSIMPL_H

#include "model/AccessPorts/SimulationAccess.h"
#include "model/context/SimulationContext.h"
#include "model/context/UnitObserver.h"
#include "model/entities/Descriptions.h"

class SimulationAccessImpl
	: public SimulationAccess
	, public UnitObserver
{
public:
	SimulationAccessImpl(QObject* parent = nullptr) : SimulationAccess(parent) {}
	virtual ~SimulationAccessImpl();

	virtual void init(SimulationContextApi* context) override;

	virtual IntVector2D getUniverseSize() const override;
	
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
