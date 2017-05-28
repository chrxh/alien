#pragma once


#include "Model/AccessPorts/SimulationAccess.h"
#include "Model/Context/SimulationContext.h"
#include "Model/Context/UnitObserver.h"
#include "Model/Entities/Descriptions.h"

class SimulationAccessGpuImpl
	: public SimulationAccess
{
public:
	SimulationAccessGpuImpl(QObject* parent = nullptr) : SimulationAccess(parent) {}
	virtual ~SimulationAccessGpuImpl() = default;

	virtual void init(SimulationContextApi* context) override;

	virtual void updateData(DataDescription const &desc) override;
	virtual void requireData(IntRect rect, ResolveDescription const& resolveDesc) override;
	virtual DataDescription const& retrieveData() override;
	
private:
	SimulationContextGpuImpl* _context = nullptr;

	DataDescription _dataCollected;
};

