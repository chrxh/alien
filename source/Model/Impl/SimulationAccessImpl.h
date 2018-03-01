#pragma once

#include "Model/Api/ChangeDescriptions.h"
#include "Model/Api/SimulationAccess.h"
#include "Model/Local/SimulationContextLocal.h"
#include "Model/Local/UnitObserver.h"

class SimulationAccessImpl
	: public SimulationAccess
	, public UnitObserver
{
public:
	SimulationAccessImpl(QObject* parent = nullptr) : SimulationAccess(parent) {}
	virtual ~SimulationAccessImpl();

	virtual void init(SimulationContext* context) override;

	virtual void clear() override;
	virtual void updateData(DataChangeDescription const &desc) override;
	virtual void requireData(IntRect rect, ResolveDescription const& resolveDesc) override;
	virtual void requireImage(IntRect rect, QImage* target) override;
	virtual DataDescription const& retrieveData() override;
	 
	//from UnitObserver
	virtual void unregister() override;
	virtual void accessToUnits() override;

private:
	void callBackClear();
	void callBackUpdateData();
	void callBackCollectData();
	void callBackDrawImage();

	void updateClusterData();
	void updateParticleData();

	void drawImageFromUnit(Unit* unit);
	void drawClustersFromUnit(Unit* unit);
	void drawParticlesFromUnit(Unit* unit);

	void collectDataFromUnit(Unit* unit);
	void collectClustersFromUnit(Unit* unit);
	void collectParticlesFromUnit(Unit* unit);

	SimulationContextLocal* _context = nullptr;
	bool _registered = false;

	bool _dataRequired = false;
	IntRect _requiredRect;
	ResolveDescription _resolveDesc;

	bool _imageRequired = false;
	QImage* _requiredImage = nullptr;

	DataDescription _dataCollected;
	DataChangeDescription _dataToUpdate;
	bool _toClear = false;
};

