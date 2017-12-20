#include <QMatrix4x4>

#include "Model/Api/SimulationAccess.h"
#include "Model/Api/SimulationContext.h"
#include "Model/Api/SimulationParameters.h"
#include "Model/Api/DescriptionHelper.h"

#include "DataRepository.h"
#include "Notifier.h"

void DataRepository::init(Notifier* notifier, SimulationAccess * access, DescriptionHelper * connector, SimulationContext* context)
{
	_descHelper = connector;
	_access = access;
	_notifier = notifier;
	_parameters = context->getSimulationParameters();

	for (auto const& connection : _connections) {
		disconnect(connection);
	}
	_connections.push_back(connect(_access, &SimulationAccess::dataReadyToRetrieve, this, &DataRepository::dataFromSimulationAvailable, Qt::QueuedConnection));
	_connections.push_back(connect(_access, &SimulationAccess::imageReady, this, &DataRepository::imageReady));
	_connections.push_back(connect(_notifier, &Notifier::notify, this, &DataRepository::sendDataChangesToSimulation));
}

DataDescription & DataRepository::getDataRef()
{
	return _data;
}

CellDescription & DataRepository::getCellDescRef(uint64_t cellId)
{
	ClusterDescription &clusterDesc = getClusterDescRef(cellId);
	int cellIndex = _navi.cellIndicesByCellIds.at(cellId);
	return clusterDesc.cells->at(cellIndex);
}

ClusterDescription & DataRepository::getClusterDescRef(uint64_t cellId)
{
	int clusterIndex = _navi.clusterIndicesByCellIds.at(cellId);
	return _data.clusters->at(clusterIndex);
}

ParticleDescription& DataRepository::getParticleDescRef(uint64_t particleId)
{
	int particleIndex = _navi.particleIndicesByParticleIds.at(particleId);
	return _data.particles->at(particleIndex);
}

void DataRepository::setSelectedTokenIndex(optional<uint> const& value)
{
	_selectedTokenIndex = value;
}

optional<uint> DataRepository::getSelectedTokenIndex() const
{
	return _selectedTokenIndex;
}

void DataRepository::addAndSelectCell(QVector2D const & posDelta)
{
	QVector2D pos = _rect.center().toQVector2D() + posDelta;
	int memorySize = _parameters->cellFunctionComputerCellMemorySize;
	auto desc = ClusterDescription().setPos(pos).setVel({}).setAngle(0).setAngularVel(0).setMetadata(ClusterMetadata()).addCell(
		CellDescription().setEnergy(_parameters->cellCreationEnergy).setMaxConnections(_parameters->cellCreationMaxConnection)
		.setPos(pos).setConnectingCells({}).setMetadata(CellMetadata())
		.setFlagTokenBlocked(false).setTokenBranchNumber(0).setCellFeature(
			CellFeatureDescription().setType(Enums::CellFunction::COMPUTER).setVolatileData(QByteArray(memorySize, 0))
		));
	_descHelper->makeValid(desc);
	_data.addCluster(desc);
	_selectedCellIds = { desc.cells->front().id };
	_selectedClusterIds = { desc.id };
	_selectedParticleIds = { };
	_navi.update(_data);
}

void DataRepository::addAndSelectParticle(QVector2D const & posDelta)
{
	QVector2D pos = _rect.center().toQVector2D() + posDelta;
	auto desc = ParticleDescription().setPos(pos).setVel({}).setEnergy(_parameters->cellMinEnergy / 2.0);
	_descHelper->makeValid(desc);
	_data.addParticle(desc);
	_selectedCellIds = { };
	_selectedClusterIds = { };
	_selectedParticleIds = { desc.id };
	_navi.update(_data);
}

namespace
{
	void correctConnections(vector<CellDescription> &cells)
	{
		unordered_set<uint64_t> cellSet;
		std::transform(cells.begin(), cells.end(), std::inserter(cellSet, cellSet.begin()),
			[](CellDescription const & cell) {
				return cell.id;
			});

		for (auto& cell : cells) {
			if (cell.connectingCells) {
				list<uint64_t> newConnectingCells;
				for (uint64_t const& connectingCell : *cell.connectingCells) {
					if (cellSet.find(connectingCell) != cellSet.end()) {
						newConnectingCells.push_back(connectingCell);
					}
				}
				cell.connectingCells = newConnectingCells;
			}
		}
	}
}


void DataRepository::deleteSelection()
{
	if (_data.clusters) {
		unordered_set<uint64_t> modifiedClusterIds;
		vector<ClusterDescription> newClusters;
		for (auto const& cluster : *_data.clusters) {
			if (_selectedClusterIds.find(cluster.id) == _selectedClusterIds.end()) {
				newClusters.push_back(cluster);
			}
			else if (cluster.cells) {
				vector<CellDescription> newCells;
				for (auto const& cell : *cluster.cells) {
					if (_selectedCellIds.find(cell.id) == _selectedCellIds.end()) {
						newCells.push_back(cell);
					}
				}
				if (!newCells.empty()) {
					correctConnections(newCells);
					ClusterDescription newCluster = cluster;
					newCluster.cells = newCells;
					newClusters.push_back(newCluster);
					modifiedClusterIds.insert(cluster.id);
				}
			}
		}
		_data.clusters = newClusters;
		if (!modifiedClusterIds.empty()) {
			_descHelper->recluster(_data, modifiedClusterIds);
		}
	}
	if (_data.particles) {
		vector<ParticleDescription> newParticles;
		for (auto const& particle : *_data.particles) {
			if (_selectedParticleIds.find(particle.id) == _selectedParticleIds.end()) {
				newParticles.push_back(particle);
			}
		}
		_data.particles = newParticles;
	}
	_selectedCellIds = {};
	_selectedClusterIds = {};
	_selectedParticleIds = {};
	_navi.update(_data);
}

void DataRepository::deleteExtendedSelection()
{
	if (_data.clusters) {
		vector<ClusterDescription> newClusters;
		for (auto const& cluster : *_data.clusters) {
			if (_selectedClusterIds.find(cluster.id) == _selectedClusterIds.end()) {
				newClusters.push_back(cluster);
			}
		}
		_data.clusters = newClusters;
	}
	if (_data.particles) {
		vector<ParticleDescription> newParticles;
		for (auto const& particle : *_data.particles) {
			if (_selectedParticleIds.find(particle.id) == _selectedParticleIds.end()) {
				newParticles.push_back(particle);
			}
		}
		_data.particles = newParticles;
	}
	_selectedCellIds = {};
	_selectedClusterIds = {};
	_selectedParticleIds = {};
	_navi.update(_data);
}

void DataRepository::addToken()
{
	CHECK(_selectedCellIds.size() == 1);
	auto& cell = getCellDescRef(*_selectedCellIds.begin());

	int numToken = cell.tokens ? cell.tokens->size() : 0;
	if (numToken < _parameters->cellMaxToken) {
		uint pos = _selectedTokenIndex ? *_selectedTokenIndex : numToken;
		cell.addToken(pos, TokenDescription().setEnergy(_parameters->tokenCreationEnergy).setData(QByteArray(_parameters->tokenMemorySize, 0)));
	}
}

void DataRepository::deleteToken()
{
	CHECK(_selectedCellIds.size() == 1);
	CHECK(_selectedTokenIndex);

	auto& cell = getCellDescRef(*_selectedCellIds.begin());
	cell.delToken(*_selectedTokenIndex);
}

bool DataRepository::isCellPresent(uint64_t cellId)
{
	return _navi.cellIds.find(cellId) != _navi.cellIds.end();
}

bool DataRepository::isParticlePresent(uint64_t particleId)
{
	return _navi.particleIds.find(particleId) != _navi.particleIds.end();
}

void DataRepository::dataFromSimulationAvailable()
{
	updateInternals(_access->retrieveData());

	Q_EMIT _notifier->notify({ Receiver::DataEditor, Receiver::VisualEditor, Receiver::Toolbar }, UpdateDescription::All);
}

void DataRepository::sendDataChangesToSimulation(set<Receiver> const& targets)
{
	if (targets.find(Receiver::Simulation) == targets.end()) {
		return;
	}
	DataChangeDescription delta(_unchangedData, _data);
	_access->updateData(delta);
	_unchangedData = _data;
}

void DataRepository::setSelection(list<uint64_t> const &cellIds, list<uint64_t> const &particleIds)
{
	_selectedCellIds = unordered_set<uint64_t>(cellIds.begin(), cellIds.end());
	_selectedParticleIds = unordered_set<uint64_t>(particleIds.begin(), particleIds.end());
	_selectedClusterIds.clear();
	for (uint64_t cellId : cellIds) {
		auto clusterIdByCellIdIter = _navi.clusterIdsByCellIds.find(cellId);
		if (clusterIdByCellIdIter != _navi.clusterIdsByCellIds.end()) {
			_selectedClusterIds.insert(clusterIdByCellIdIter->second);
		}
	}
}

bool DataRepository::isInSelection(list<uint64_t> const & ids) const
{
	for (uint64_t id : ids) {
		if (!isInSelection(id)) {
			return false;
		}
	}
	return true;
}

bool DataRepository::isInSelection(uint64_t id) const
{
	return (_selectedCellIds.find(id) != _selectedCellIds.end() || _selectedParticleIds.find(id) != _selectedParticleIds.end());
}

bool DataRepository::isInExtendedSelection(uint64_t id) const
{
	auto clusterIdByCellIdIter = _navi.clusterIdsByCellIds.find(id);
	if (clusterIdByCellIdIter != _navi.clusterIdsByCellIds.end()) {
		uint64_t clusterId = clusterIdByCellIdIter->second;
		return (_selectedClusterIds.find(clusterId) != _selectedClusterIds.end() || _selectedParticleIds.find(id) != _selectedParticleIds.end());
	}
	return false;
}

bool DataRepository::areEntitiesSelected() const
{
	return !_selectedCellIds.empty() || !_selectedParticleIds.empty();
}

unordered_set<uint64_t> DataRepository::getSelectedCellIds() const
{
	return _selectedCellIds;
}

unordered_set<uint64_t> DataRepository::getSelectedParticleIds() const
{
	return _selectedParticleIds;
}

void DataRepository::moveSelection(QVector2D const &delta)
{
	for (uint64_t cellId : _selectedCellIds) {
		if (isCellPresent(cellId)) {
			int clusterIndex = _navi.clusterIndicesByCellIds.at(cellId);
			int cellIndex = _navi.cellIndicesByCellIds.at(cellId);
			CellDescription &cellDesc = getCellDescRef(cellId);
			cellDesc.pos = *cellDesc.pos + delta;
		}
	}

	for (uint64_t particleId : _selectedParticleIds) {
		if (isParticlePresent(particleId)) {
			ParticleDescription &particleDesc = getParticleDescRef(particleId);
			particleDesc.pos = *particleDesc.pos + delta;
		}
	}
}

void DataRepository::moveExtendedSelection(QVector2D const & delta)
{
	for (uint64_t selectedClusterId : _selectedClusterIds) {
		auto selectedClusterIndex = _navi.clusterIndicesByClusterIds.at(selectedClusterId);
		ClusterDescription &clusterDesc = _data.clusters->at(selectedClusterIndex);
		clusterDesc.pos = *clusterDesc.pos + delta;
	}

	list<uint64_t> extSelectedCellIds;
	for (auto clusterIdByCellId : _navi.clusterIdsByCellIds) {
		uint64_t cellId = clusterIdByCellId.first;
		uint64_t clusterId = clusterIdByCellId.second;
		if (_selectedClusterIds.find(clusterId) != _selectedClusterIds.end()) {
			extSelectedCellIds.push_back(cellId);
		}
	}

	for (uint64_t cellId : extSelectedCellIds) {
		if (isCellPresent(cellId)) {
			int clusterIndex = _navi.clusterIndicesByCellIds.at(cellId);
			int cellIndex = _navi.cellIndicesByCellIds.at(cellId);
			CellDescription &cellDesc = getCellDescRef(cellId);
			cellDesc.pos = *cellDesc.pos + delta;
		}
	}

	for (uint64_t particleId : _selectedParticleIds) {
		if (isParticlePresent(particleId)) {
			ParticleDescription &particleDesc = getParticleDescRef(particleId);
			particleDesc.pos = *particleDesc.pos + delta;
		}
	}
}

void DataRepository::reconnectSelectedCells()
{
	_descHelper->reconnect(getDataRef(), getSelectedCellIds());
	updateAfterCellReconnections();
}

void DataRepository::rotateSelection(double angle)
{
	QVector3D center = calcCenter();

	QMatrix4x4 transform;
	transform.setToIdentity();
	transform.translate(center);
	transform.rotate(angle, 0.0, 0.0, 1.0);
	transform.translate(-center);

	if (_data.clusters) {
		for (uint64_t clusterId : _selectedClusterIds) {
			auto& cluster = _data.clusters->at(_navi.clusterIndicesByClusterIds.at(clusterId));
			if (!cluster.cells) {
				continue;
			}
			for (auto& cell : *cluster.cells) {
				*cell.pos = transform.map(QVector3D(*cell.pos)).toVector2D();
			}
			*cluster.angle += angle;
			*cluster.pos = transform.map(QVector3D(*cluster.pos)).toVector2D();
		}
	}
	if (_data.particles) {
		for (uint64_t particleId : _selectedParticleIds) {
			auto& particle = getParticleDescRef(particleId);
			*particle.pos = transform.map(QVector3D(*particle.pos)).toVector2D();
		}
	}
}

QVector2D DataRepository::calcCenter()
{
	QVector2D result;
	int numEntities = 0;
	if (_data.clusters) {
		for (uint64_t clusterId : _selectedClusterIds) {
			auto const& cluster = _data.clusters->at(_navi.clusterIndicesByClusterIds.at(clusterId));
			if (!cluster.cells) {
				continue;
			}
			for (auto const& cell : *cluster.cells) {
				result += *cell.pos;
				++numEntities;
			}
		}
	}
	if (_data.particles) {
		for (uint64_t particleId: _selectedParticleIds) {
			auto const& particle = getParticleDescRef(particleId);
			result += *particle.pos;
			++numEntities;
		}
	}
	CHECK(numEntities > 0);
	result /= numEntities;
	return result;
}

void DataRepository::updateCluster(ClusterDescription const & cluster)
{
	int clusterIndex = _navi.clusterIndicesByClusterIds.at(cluster.id);
	_data.clusters->at(clusterIndex) = cluster;

	_navi.update(_data);
}

void DataRepository::updateParticle(ParticleDescription const & particle)
{
	int particleIndex = _navi.particleIndicesByParticleIds.at(particle.id);
	_data.particles->at(particleIndex) = particle;

	_navi.update(_data);
}

void DataRepository::requireDataUpdateFromSimulation(IntRect const& rect)
{
	_rect = rect;
	ResolveDescription resolveDesc;
	resolveDesc.resolveCellLinks = true;
	_access->requireData(rect, resolveDesc);
}

void DataRepository::requireImageFromSimulation(IntRect const & rect, QImage * target)
{
	_access->requireImage(rect, target);
}

void DataRepository::updateAfterCellReconnections()
{
	_navi.update(_data);

	_selectedClusterIds.clear();
	for (uint64_t selectedCellId : _selectedCellIds) {
		if (_navi.clusterIdsByCellIds.find(selectedCellId) != _navi.clusterIdsByCellIds.end()) {
			_selectedClusterIds.insert(_navi.clusterIdsByCellIds.at(selectedCellId));
		}
	}
}

void DataRepository::updateInternals(DataDescription const &data)
{
	_data = data;
	_unchangedData = _data;

	_navi.update(data);

	unordered_set<uint64_t> newSelectedCells;
	std::copy_if(_selectedCellIds.begin(), _selectedCellIds.end(), std::inserter(newSelectedCells, newSelectedCells.begin()), 
		[this](uint64_t cellId) {
			return _navi.cellIds.find(cellId) != _navi.cellIds.end();
		});
	_selectedCellIds = newSelectedCells;

	unordered_set<uint64_t> newSelectedClusterIds;
	std::copy_if(_selectedClusterIds.begin(), _selectedClusterIds.end(), std::inserter(newSelectedClusterIds, newSelectedClusterIds.begin()),
		[this](uint64_t clusterId) {
			return _navi.clusterIndicesByClusterIds.find(clusterId) != _navi.clusterIndicesByClusterIds.end();
		});
	_selectedClusterIds = newSelectedClusterIds;

	unordered_set<uint64_t> newSelectedParticles;
	std::copy_if(_selectedParticleIds.begin(), _selectedParticleIds.end(), std::inserter(newSelectedParticles, newSelectedParticles.begin()),
		[this](uint64_t particleId) {
			return _navi.particleIds.find(particleId) != _navi.particleIds.end();
		});
	_selectedParticleIds = newSelectedParticles;
}

