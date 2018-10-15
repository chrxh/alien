#include <QImage>

#include "ModelBasic/SpaceProperties.h"

#include "CudaBridge.h"
#include "ThreadController.h"
#include "SimulationContextGpuImpl.h"
#include "SimulationAccessGpuImpl.h"
#include "SimulationControllerGpu.h"

SimulationAccessGpuImpl::~SimulationAccessGpuImpl()
{
}

void SimulationAccessGpuImpl::init(SimulationControllerGpu* controller)
{
	_context = static_cast<SimulationContextGpuImpl*>(controller->getContext());
	auto worker = _context->getGpuThreadController()->getGpuWorker();
	connect(worker, &CudaBridge::dataReadyToRetrieve, this, &SimulationAccessGpuImpl::dataReadyToRetrieveFromGpu);
}

void SimulationAccessGpuImpl::clear()
{
}

void SimulationAccessGpuImpl::updateData(DataChangeDescription const & desc)
{
}

void SimulationAccessGpuImpl::requireData(IntRect rect, ResolveDescription const & resolveDesc)
{
	_dataRequired = true;
	_requiredRect = rect;
	_resolveDesc = resolveDesc;

	auto worker = _context->getGpuThreadController()->getGpuWorker();
	worker->requireData();
}

void SimulationAccessGpuImpl::requireImage(IntRect rect, QImage * target)
{
	_imageRequired = true;
	_requiredRect = rect;
	_requiredImage = target;

	auto worker = _context->getGpuThreadController()->getGpuWorker();
	worker->requireData();
}

DataDescription const & SimulationAccessGpuImpl::retrieveData()
{
	return _dataCollected;
}

void SimulationAccessGpuImpl::dataReadyToRetrieveFromGpu()
{

	if (_imageRequired) {
		_imageRequired = false;
		createImage();
		Q_EMIT imageReady();

	}

	if (_dataRequired) {
		_dataRequired = false;
		createData();
		Q_EMIT dataReadyToRetrieve();
	}
}

void SimulationAccessGpuImpl::createImage()
{
	auto metric = _context->getSpaceProperties();
	auto worker = _context->getGpuThreadController()->getGpuWorker();
	DataForAccess cudaData = worker->retrieveData();

	_requiredImage->fill(QColor(0x00, 0x00, 0x1b));

	worker->lockData();

	for (int i = 0; i < cudaData.numParticles; ++i) {
		ParticleData& particle = cudaData.particles[i];
		float2& pos = particle.pos;
		IntVector2D intPos = { static_cast<int>(pos.x), static_cast<int>(pos.y) };
		if (_requiredRect.isContained(intPos)) {
			metric->correctPosition(intPos);
			_requiredImage->setPixel(intPos.x, intPos.y, 0x902020);
		}
	}

	for (int i = 0; i < cudaData.numCells; ++i) {
		CellData& cell = cudaData.cells[i];
		float2& pos = cell.absPos;
		IntVector2D intPos = { static_cast<int>(pos.x), static_cast<int>(pos.y) };
		if (_requiredRect.isContained(intPos)) {
			metric->correctPosition(intPos);
			_requiredImage->setPixel(intPos.x, intPos.y, 0xFF);
		}
	}

	worker->unlockData();
}

void SimulationAccessGpuImpl::createData()
{
	_dataCollected.clear();
	auto worker = _context->getGpuThreadController()->getGpuWorker();
	DataForAccess cudaData = worker->retrieveData();

	worker->lockData();

	list<uint64_t> connectingCellIds;
	for (int i = 0; i < cudaData.numClusters; ++i) {
		ClusterData const& cluster = cudaData.clusters[i];
		if (_requiredRect.isContained({ int(cluster.pos.x), int(cluster.pos.y) })) {
			auto clusterDesc = ClusterDescription().setPos({ cluster.pos.x, cluster.pos.y })
				.setVel({ cluster.vel.x, cluster.vel.y })
				.setAngle(cluster.angle)
				.setAngularVel(cluster.angularVel).setMetadata(ClusterMetadata());

			for (int j = 0; j < cluster.numCells; ++j) {
				CellData const& cell = cluster.cells[j];
				auto pos = cell.absPos;
				auto id = cell.id;
				connectingCellIds.clear();
				for (int i = 0; i < cell.numConnections; ++i) {
					connectingCellIds.emplace_back(cell.connections[i]->id);
				}
				clusterDesc.addCell(
					CellDescription().setPos({ pos.x, pos.y }).setMetadata(CellMetadata())
					.setEnergy(100.0f).setId(id).setCellFeature(CellFeatureDescription().setType(Enums::CellFunction::COMPUTER))
					.setConnectingCells(connectingCellIds).setMaxConnections(CELL_MAX_BONDS).setFlagTokenBlocked(false)
					.setTokenBranchNumber(0).setMetadata(CellMetadata())
				);
			}
			_dataCollected.addCluster(clusterDesc);
		}
	}

	worker->unlockData();
}

