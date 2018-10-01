#include "ModelInterface/ChangeDescriptions.h"

#include "DataEditModel.h"
#include "Gui/DataRepository.h"

DataEditModel::DataEditModel(QObject *parent)
	: QObject(parent)
{
}

void DataEditModel::init(DataRepository* manipulator, SimulationParameters const* parameters, SymbolTable* symbols)
{
	_manipulator = manipulator;
	_parameters = parameters;
	_symbols = symbols;
}

void DataEditModel::setClusterAndCell(ClusterDescription const & cluster, uint64_t cellId)
{
	_data.clear();
	_data.addCluster(cluster);
	_unchangedData = _data;
	_navi.update(_data);

	_selectedCellIds = { cellId };
	_selectedParticleIds.clear();
}

void DataEditModel::setParticle(ParticleDescription const & particle)
{
	_data.clear();
	_data.addParticle(particle);
	_unchangedData = _data;
	_navi.update(_data);

	_selectedCellIds.clear();
	_selectedParticleIds = { particle.id };
}

void DataEditModel::setSelectionIds(unordered_set<uint64_t> const& selectedCellIds, unordered_set<uint64_t> const& selectedParticleIds)
{
	_data.clear();
	_unchangedData = _data;
	_navi.update(_data);

	_selectedCellIds = selectedCellIds;
	_selectedParticleIds = selectedParticleIds;
}

void DataEditModel::setSelectedTokenIndex(optional<uint> const & value)
{
	_manipulator->setSelectedTokenIndex(value);
}

optional<uint> DataEditModel::getSelectedTokenIndex() const
{
	return _manipulator->getSelectedTokenIndex();
}

DataChangeDescription DataEditModel::getAndUpdateChanges()
{
	DataChangeDescription result(_unchangedData, _data);
	_unchangedData = _data;
	return result;
}

TokenDescription & DataEditModel::getTokenToEditRef(int tokenIndex)
{
	auto& cell = getCellToEditRef();
	return cell.tokens->at(tokenIndex);
}

CellDescription & DataEditModel::getCellToEditRef()
{
	uint64_t selectedCellId = *_selectedCellIds.begin();
	int clusterIndex = _navi.clusterIndicesByCellIds.at(selectedCellId);
	int cellIndex = _navi.cellIndicesByCellIds.at(selectedCellId);
	return _data.clusters->at(clusterIndex).cells->at(cellIndex);
}

ParticleDescription & DataEditModel::getParticleToEditRef()
{
	uint64_t selectedParticleId = *_selectedParticleIds.begin();
	int particleIndex = _navi.particleIndicesByParticleIds.at(selectedParticleId);
	return _data.particles->at(particleIndex);
}

ClusterDescription & DataEditModel::getClusterToEditRef()
{
	uint64_t selectedCellId = *_selectedCellIds.begin();
	int clusterIndex = _navi.clusterIndicesByCellIds.at(selectedCellId);
	return _data.clusters->at(clusterIndex);
}

int DataEditModel::getNumCells() const
{
	return _selectedCellIds.size();
}

int DataEditModel::getNumParticles() const
{
	return _selectedParticleIds.size();
}

SimulationParameters const* DataEditModel::getSimulationParameters() const
{
	return _parameters;
}

SymbolTable * DataEditModel::getSymbolTable() const
{
	return _symbols;
}

void DataEditModel::setSymbolTable(SymbolTable * symbols)
{
	_symbols = symbols;
}

