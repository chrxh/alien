#pragma once

#include "ModelBasic/ChangeDescriptions.h"
#include "SimulationAccessCpu.h"
#include "SimulationContextImpl.h"
#include "UnitObserver.h"

class SimulationAccessCpuImpl
	: public SimulationAccessCpu
	, public UnitObserver
{
public:
	SimulationAccessCpuImpl(QObject* parent = nullptr) : SimulationAccessCpu(parent) {}
	virtual ~SimulationAccessCpuImpl();

	virtual void init(SimulationControllerCpu* controller) override;

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

	SimulationContextImpl* _context = nullptr;
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

