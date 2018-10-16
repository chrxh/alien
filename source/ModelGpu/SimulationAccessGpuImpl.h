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
	Q_SLOT void dataRequiredFromGpu();
	void updateDataToGpuModel();
	void createImageFromGpuModel();
	void createDataFromGpuModel();

	SimulationContextGpuImpl* _context = nullptr;

	bool _dataUpdate = false;
	DataChangeDescription _dataToUpdate;

	bool _dataDescRequired = false;
	ResolveDescription _resolveDesc;	//not used yet
	DataDescription _dataCollected;

	bool _imageRequired = false;
	IntRect _requiredRect;
	QImage* _requiredImage = nullptr;


};

