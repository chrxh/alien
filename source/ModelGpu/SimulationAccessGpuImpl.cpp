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
	auto cudaBridge = _context->getGpuThreadController()->getCudaBridge();
	connect(cudaBridge, &CudaBridge::dataAccessGranted, this, &SimulationAccessGpuImpl::dataAccessGrantedFromGpu);
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

	auto cudaBridge = _context->getGpuThreadController()->getCudaBridge();
	cudaBridge->requireAccess();
}

void SimulationAccessGpuImpl::requireImage(IntRect rect, QImage * target)
{
	_imageRequired = true;
	_requiredRect = rect;
	_requiredImage = target;

	auto worker = _context->getGpuThreadController()->getCudaBridge();
	worker->requireAccess();
}

DataDescription const & SimulationAccessGpuImpl::retrieveData()
{
	return _dataCollected;
}

void SimulationAccessGpuImpl::dataAccessGrantedFromGpu()
{
	if (_dataUpdate) {
		_dataUpdate = false;
		updateDataToGpuModel();
	}

	if (_imageRequired) {
		_imageRequired = false;
		createImageFromGpuModel();
		Q_EMIT imageReady();

	}

	if (_dataRequired) {
		_dataRequired = false;
		createDataFromGpuModel();
		Q_EMIT dataReadyToRetrieve();
	}
}

void SimulationAccessGpuImpl::updateDataToGpuModel()
{
	auto cudaBridge = _context->getGpuThreadController()->getCudaBridge();
	SimulationDataForAccess cudaData = cudaBridge->retrieveData();

	cudaBridge->lockData();

	cudaBridge->unlockData();
}

void SimulationAccessGpuImpl::createImageFromGpuModel()
{
	auto spaceProp = _context->getSpaceProperties();
	auto worker = _context->getGpuThreadController()->getCudaBridge();
	SimulationDataForAccess cudaData = worker->retrieveData();

	_requiredImage->fill(QColor(0x00, 0x00, 0x1b));

	worker->lockData();

	for (int i = 0; i < cudaData.numParticles; ++i) {
		ParticleData& particle = cudaData.particles[i];
		float2& pos = particle.pos;
		IntVector2D intPos = { static_cast<int>(pos.x), static_cast<int>(pos.y) };
		if (_requiredRect.isContained(intPos)) {
			spaceProp->correctPosition(intPos);
			_requiredImage->setPixel(intPos.x, intPos.y, 0x902020);
		}
	}

	for (int i = 0; i < cudaData.numCells; ++i) {
		CellData& cell = cudaData.cells[i];
		float2& pos = cell.absPos;
		IntVector2D intPos = { static_cast<int>(pos.x), static_cast<int>(pos.y) };
		if (_requiredRect.isContained(intPos)) {
			spaceProp->correctPosition(intPos);
			_requiredImage->setPixel(intPos.x, intPos.y, 0xFF);
		}
	}

	worker->unlockData();
}

void SimulationAccessGpuImpl::createDataFromGpuModel()
{
	_dataCollected.clear();
	auto cudaBridge = _context->getGpuThreadController()->getCudaBridge();
	SimulationDataForAccess cudaData = cudaBridge->retrieveData();

	cudaBridge->lockData();

	list<uint64_t> connectingCellIds;
	for (int i = 0; i < cudaData.numClusters; ++i) {
		ClusterData const& cluster = cudaData.clusters[i];
		if (_requiredRect.isContained({ int(cluster.pos.x), int(cluster.pos.y) })) {
			auto clusterDesc = ClusterDescription().setId(cluster.id).setPos({ cluster.pos.x, cluster.pos.y })
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

	for (int i = 0; i < cudaData.numParticles; ++i) {
		ParticleData const& particle = cudaData.particles[i];
		if (_requiredRect.isContained({ int(particle.pos.x), int(particle.pos.y) })) {
			_dataCollected.addParticle(ParticleDescription().setId(particle.id).setPos({ particle.pos.x, particle.pos.y })
				.setVel({ particle.vel.x, particle.vel.y }).setEnergy(particle.energy));
		}
	}

	cudaBridge->unlockData();
}

