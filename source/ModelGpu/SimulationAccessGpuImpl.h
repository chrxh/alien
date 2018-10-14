#pragma once

#include "ModelBasic/SimulationAccess.h"
#include "ModelBasic/ChangeDescriptions.h"

#include "SimulationAccessGpu.h"

class SimulationAccessGpuImpl
	: public SimulationAccessGpu
{
public:
	SimulationAccessGpuImpl(QObject* parent = nullptr) : SimulationAccessGpu(parent) {}
	virtual ~SimulationAccessGpuImpl();

	virtual void init(SimulationControllerGpu* controller) override;

	virtual void clear() override;
	virtual void updateData(DataChangeDescription const &desc) override;
	virtual void requireData(IntRect rect, ResolveDescription const& resolveDesc) override;
	virtual void requireImage(IntRect rect, QImage* target) override;
	virtual DataDescription const& retrieveData() override;

private:
	Q_SLOT void dataReadyToRetrieveFromGpu();
	void createImage();
	void createData();

	SimulationContextGpuImpl* _context = nullptr;

	bool _dataRequired = false;
	bool _imageRequired = false;

	IntRect _requiredRect;
	ResolveDescription _resolveDesc;
	QImage* _requiredImage = nullptr;

	DataDescription _dataCollected;
};

