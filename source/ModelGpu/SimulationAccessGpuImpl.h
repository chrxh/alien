#pragma once

#include "ModelBasic/SimulationAccess.h"
#include "ModelBasic/ChangeDescriptions.h"

#include "CudaConstants.h"
#include "SimulationAccessGpu.h"

class SimulationAccessGpuImpl
	: public SimulationAccessGpu
{
public:
	SimulationAccessGpuImpl(QObject* parent = nullptr);
	virtual ~SimulationAccessGpuImpl();

	void init(SimulationControllerGpu* controller) override;

	void clear() override;
	void updateData(DataChangeDescription const &dataToUpdate) override;
    void requireData(ResolveDescription const& resolveDesc) override;
    void requireData(IntRect rect, ResolveDescription const& resolveDesc) override;
	void requirePixelImage(IntRect rect, QImagePtr const& target, std::mutex& mutex) override;
    void requireVectorImage(IntRect rect, double zoom, QImagePtr const& target, std::mutex& mutex) override;
    void selectEntities(IntVector2D const& pos) override;
    void deselectAll() override;
    void applyAction(PhysicalAction const& action) override;
    DataDescription const& retrieveData() override;

private:
    void scheduleJob(CudaJob const& job);
	Q_SLOT void jobsFinished();

	void createDataFromGpuModel(DataAccessTO dataTO, IntRect const& rect);

	void metricCorrection(DataChangeDescription& data) const;

	string getObjectId() const;

	class _DataTOCache
	{
	public:
		_DataTOCache(CudaConstants const& cudaConstants);
		~_DataTOCache();

		DataAccessTO getDataTO();
		void releaseDataTO(DataAccessTO const& dataTO);

	private:
        DataAccessTO getNewDataTO();
        void deleteDataTO(DataAccessTO const& dataTO);

        CudaConstants _cudaConstants;
        vector<DataAccessTO> _freeDataTOs;
		vector<DataAccessTO> _usedDataTOs;
	};
    using DataTOCache = boost::shared_ptr<_DataTOCache>;

private:
	list<QMetaObject::Connection> _connections;

	SimulationContextGpuImpl* _context = nullptr;
	NumberGenerator* _numberGen = nullptr;
    CudaConstants _cudaConstants;

	DataDescription _dataCollected;
	DataTOCache _dataTOCache;
	IntRect _lastDataRect;
};

