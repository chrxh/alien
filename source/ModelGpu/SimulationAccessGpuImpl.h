#pragma once

#include "ModelBasic/SimulationAccess.h"
#include "ModelBasic/ChangeDescriptions.h"

#include "SimulationAccessGpu.h"

class SimulationAccessGpuImpl
	: public SimulationAccessGpu
{
public:
	SimulationAccessGpuImpl(QObject* parent = nullptr);
	virtual ~SimulationAccessGpuImpl();

	virtual void init(SimulationControllerGpu* controller) override;

	virtual void clear() override;
	virtual void updateData(DataChangeDescription const &dataToUpdate) override;
	virtual void requireData(IntRect rect, ResolveDescription const& resolveDesc) override;
	virtual void requireImage(IntRect rect, QImage* target) override;
	virtual DataDescription const& retrieveData() override;

private:
	Q_SLOT void jobsFinished();

	void updateDataToGpu(DataAccessTO dataToUpdateTO, IntRect const& rect, DataChangeDescription const& updateDesc);
	void createImageFromGpuModel(DataAccessTO const& dataTO, IntRect const& rect, QImage* targetImage);
	void createDataFromGpuModel(DataAccessTO dataTO, IntRect const& rect);

	void metricCorrection(DataChangeDescription& data) const;

	string getObjectId() const;

	class DataTOCache
	{
	public:
		DataTOCache();
		~DataTOCache();

		DataAccessTO getDataTO();
		void releaseDataTO(DataAccessTO const& dataTO);

	private:
		vector<DataAccessTO> _freeDataTOs;
		vector<DataAccessTO> _usedDataTOs;
	};

private:
	list<QMetaObject::Connection> _connections;

	SimulationContextGpuImpl* _context = nullptr;
	NumberGenerator* _numberGen = nullptr;

	DataDescription _dataCollected;
	DataTOCache _dataTOCache;
	IntRect _lastDataRect;

	bool _updateInProgress = false;
	vector<CudaJob> _waitingJobs;
};

