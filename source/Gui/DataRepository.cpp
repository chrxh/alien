#include <QMatrix4x4>

#include "Base/NumberGenerator.h"

#include "Model/Api/SimulationAccess.h"
#include "Model/Api/SimulationContext.h"
#include "Model/Api/SimulationParameters.h"
#include "Model/Api/DescriptionHelper.h"
#include "Model/Api/SpaceProperties.h"

#include "DataRepository.h"
#include "Notifier.h"

void DataRepository::init(Notifier* notifier, SimulationAccess * access, DescriptionHelper * connector
	, SimulationContext* context, NumberGenerator* numberGenerator)
{
	_descHelper = connector;
	_access = access;
	_notifier = notifier;
	_numberGenerator = numberGenerator;
	_parameters = context->getSimulationParameters();
	_universeSize = context->getSpaceProperties()->getSize();
	_unchangedData.clear();
	_data.clear();
	_selectedCellIds.clear();
	_selectedClusterIds.clear();
	_selectedParticleIds.clear();
	_selectedTokenIndex.reset();
	
	for (auto const& connection : _connections) {
		disconnect(connection);
	}
	_connections.push_back(connect(_access, &SimulationAccess::dataReadyToRetrieve, this, &DataRepository::dataFromSimulationAvailable, Qt::QueuedConnection));
	_connections.push_back(connect(_access, &SimulationAccess::imageReady, this, &DataRepository::imageReady));
	_connections.push_back(connect(_notifier, &Notifier::notifyDataRepositoryChanged, this, &DataRepository::sendDataChangesToSimulation));
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

ClusterDescription const & DataRepository::getClusterDescRef(uint64_t cellId) const
{
	int clusterIndex = _navi.clusterIndicesByCellIds.at(cellId);
	return _data.clusters->at(clusterIndex);
}

ParticleDescription& DataRepository::getParticleDescRef(uint64_t particleId)
{
	int particleIndex = _navi.particleIndicesByParticleIds.at(particleId);
	return _data.particles->at(particleIndex);
}

ParticleDescription const & DataRepository::getParticleDescRef(uint64_t particleId) const
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
		CellDescription().setEnergy(_parameters->cellFunctionConstructorOffspringCellEnergy).setMaxConnections(_parameters->cellCreationMaxConnection)
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

void DataRepository::addAndSelectData(DataDescription data, QVector2D const & posDelta)
{
	QVector2D centerOfData = data.calcCenter();
	QVector2D targetCenter = _rect.center().toQVector2D() + posDelta;
	QVector2D delta = targetCenter - centerOfData;
	data.shift(delta);

	_selectedCellIds = {};
	_selectedClusterIds = {};
	_selectedParticleIds = {};
	if (data.clusters) {
		for (auto& cluster : *data.clusters) {
			cluster.id = 0;
			_descHelper->makeValid(cluster);
			_data.addCluster(cluster);
			_selectedClusterIds.insert(cluster.id);
			if (cluster.cells) {
				std::transform(cluster.cells->begin(), cluster.cells->end(), std::inserter(_selectedCellIds, _selectedCellIds.begin())
					, [](auto const& cell) { return cell.id; });
			}
		}
	}
	if (data.particles) {
		for (auto& particle : *data.particles) {
			particle.id = 0;
			_descHelper->makeValid(particle);
			_data.addParticle(particle);
			_selectedParticleIds.insert(particle.id);
		}
	}
	_navi.update(_data);
}

namespace
{
	QVector2D calcCenter(int numCluster, int numParticles, std::function<ClusterDescription&(int)> clusterResolver
		, std::function<ParticleDescription&(int)> particleResolver)
	{
		QVector2D result;
		int numEntities = 0;
		for (int i = 0; i < numCluster; ++i) {
			auto const& cluster = clusterResolver(i);
			if (!cluster.cells) {
				continue;
			}
			for (auto const& cell : *cluster.cells) {
				result += *cell.pos;
				++numEntities;
			}
		}
		for (int i = 0; i < numParticles; ++i) {
			auto const& particle = particleResolver(i);
			result += *particle.pos;
			++numEntities;
		}
		CHECK(numEntities > 0);
		result /= numEntities;
		return result;
	}

	void rotate(double angle, int numCluster, int numParticles, std::function<ClusterDescription&(int)> clusterResolver
		, std::function<ParticleDescription&(int)> particleResolver)
	{
		QVector3D center = calcCenter(numCluster, numParticles, clusterResolver, particleResolver);

		QMatrix4x4 transform;
		transform.setToIdentity();
		transform.translate(center);
		transform.rotate(angle, 0.0, 0.0, 1.0);
		transform.translate(-center);

		for (int i = 0; i < numCluster; ++i) {
			auto& cluster = clusterResolver(i);
			if (!cluster.cells) {
				continue;
			}
			for (auto& cell : *cluster.cells) {
				*cell.pos = transform.map(QVector3D(*cell.pos)).toVector2D();
			}
			*cluster.angle += angle;
			*cluster.pos = transform.map(QVector3D(*cluster.pos)).toVector2D();
		}
		for (int i = 0; i < numParticles; ++i) {
			auto& particle = particleResolver(i);
			*particle.pos = transform.map(QVector3D(*particle.pos)).toVector2D();
		}

	}
}

void DataRepository::addDataAtFixedPosition(DataDescription data, optional<double> rotationAngle)
{
	if (rotationAngle) {
		int numClusters = data.clusters.is_initialized() ? data.clusters->size() : 0;
		int numParticle = data.particles.is_initialized() ? data.particles->size() : 0;
		auto clusterResolver = [&data](int index) -> ClusterDescription& {
			return data.clusters->at(index);
		};
		auto particleResolver = [&data](int index) -> ParticleDescription& {
			return data.particles->at(index);
		};
		rotate(*rotationAngle, numClusters, numParticle, clusterResolver, particleResolver);
	}

	if (data.clusters) {
		for (auto& cluster : *data.clusters) {
			cluster.id = 0;
			_descHelper->makeValid(cluster);
			_data.addCluster(cluster);
		}
	}
	if (data.particles) {
		for (auto& particle : *data.particles) {
			particle.id = 0;
			_descHelper->makeValid(particle);
			_data.addParticle(particle);
		}
	}
	_navi.update(_data);
}

void DataRepository::addRandomParticles(double totalEnergy, double maxEnergyPerParticle)
{
	DataDescription data;
	double remainingEnergy = totalEnergy;
	while (remainingEnergy > FLOATINGPOINT_MEDIUM_PRECISION) {
		double particleEnergy = _numberGenerator->getRandomReal(maxEnergyPerParticle / 100.0, maxEnergyPerParticle);
		particleEnergy = std::min(particleEnergy, remainingEnergy);
		data.addParticle(ParticleDescription()
			.setPos(QVector2D(_numberGenerator->getRandomReal(0.0, _universeSize.x), _numberGenerator->getRandomReal(0.0, _universeSize.y)))
			.setVel(QVector2D(_numberGenerator->getRandomReal()*2.0 - 1.0, _numberGenerator->getRandomReal()*2.0 - 1.0))
			.setEnergy(particleEnergy));
		remainingEnergy -= particleEnergy;
	}

	addDataAtFixedPosition(data);
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
	addToken(TokenDescription().setEnergy(_parameters->cellFunctionConstructorOffspringTokenEnergy).setData(QByteArray(_parameters->tokenMemorySize, 0)));
}

void DataRepository::addToken(TokenDescription const & token)
{
	CHECK(_selectedCellIds.size() == 1);
	auto& cell = getCellDescRef(*_selectedCellIds.begin());

	int numToken = cell.tokens ? cell.tokens->size() : 0;
	if (numToken < _parameters->cellMaxToken) {
		uint pos = _selectedTokenIndex ? *_selectedTokenIndex : numToken;
		cell.addToken(pos, token);
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

	Q_EMIT _notifier->notifyDataRepositoryChanged({ Receiver::DataEditor, Receiver::VisualEditor, Receiver::ActionController }, UpdateDescription::All);
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

namespace
{
	template<typename T>
	unordered_set<T> calcDifference(unordered_set<T> const& set1, unordered_set<T> const& set2)
	{
		unordered_set<T> result;
		std::set_difference(set1.begin(), set1.end(), set2.begin(), set2.begin(), std::inserter(result, result.begin()));
		return result;
	}
}

unordered_set<uint64_t> DataRepository::getSelectedCellIds() const
{
	return calcDifference<uint64_t>(_selectedCellIds, _navi.cellIds);
}

unordered_set<uint64_t> DataRepository::getSelectedParticleIds() const
{
	return calcDifference<uint64_t>(_selectedParticleIds, _navi.particleIds);
}

DataDescription DataRepository::getExtendedSelection() const
{
	DataDescription result;
	for (uint64_t clusterId : _selectedClusterIds) {
		int clusterIndex = _navi.clusterIndicesByClusterIds.at(clusterId);
		result.addCluster(_data.clusters->at(clusterIndex));
	}
	for (uint64_t particleId : _selectedParticleIds) {
		result.addParticle(getParticleDescRef(particleId));
	}
	return result;
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
	_descHelper->reconnect(getDataRef(), _unchangedData, getSelectedCellIds());
	updateAfterCellReconnections();
}

void DataRepository::rotateSelection(double angle)
{
	vector<uint64_t> selectedClusterIds(_selectedClusterIds.begin(), _selectedClusterIds.end());
	vector<uint64_t> selectedParticleIds(_selectedParticleIds.begin(), _selectedParticleIds.end());
	auto clusterResolver = [&selectedClusterIds, this](int index) -> ClusterDescription&  {
		return _data.clusters->at(_navi.clusterIndicesByClusterIds.at(selectedClusterIds.at(index)));
	};
	auto particleResolver = [&selectedParticleIds, this](int index) -> ParticleDescription& {
		return getParticleDescRef(selectedParticleIds.at(index));
	};
	rotate(angle, _selectedClusterIds.size(), _selectedParticleIds.size(), clusterResolver, particleResolver);
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
	_rect = rect;
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

