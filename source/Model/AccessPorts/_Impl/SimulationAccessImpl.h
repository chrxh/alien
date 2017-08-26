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

	virtual void updateData(DataChangeDescription const &desc) override;
	virtual void requireData(IntRect rect, ResolveDescription const& resolveDesc) override;
	virtual void requireImage(IntRect rect, QImage* target) override;
	virtual DataChangeDescription const& retrieveData() override;

	virtual void unregister() override;
	virtual void accessToUnits() override;
private:
	void callBackUpdateData();
	void callBackCollectData();
	void callBackDrawImage();

	void drawImageFromUnit(Unit* unit);
	void drawClustersFromUnit(Unit* unit);
	void drawParticlesFromUnit(Unit* unit);

	void collectDataFromUnit(Unit* unit);
	void collectClustersFromUnit(Unit* unit);
	void collectParticlesFromUnit(Unit* unit);

	SimulationContext* _context = nullptr;
	bool _registered = false;

	bool _dataRequired = false;
	IntRect _requiredRect;
	ResolveDescription _resolveDesc;

	bool _imageRequired = false;
	QImage* _requiredImage = nullptr;

	DataChangeDescription _dataCollected;
	DataChangeDescription _dataToUpdate;
};

#endif // SIMULATIONACCESSIMPL_H
