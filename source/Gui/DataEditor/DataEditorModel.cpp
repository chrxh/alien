#include "Model/Api/ChangeDescriptions.h"

#include "DataEditorModel.h"

DataEditorModel::DataEditorModel(QObject *parent)
	: QObject(parent)
{
}

void DataEditorModel::setClusterAndCell(ClusterDescription const & cluster, uint64_t cellId)
{
	_data.clear();
	_data.addCluster(cluster);
	_unchangedData = _data;
	_navi.update(_data);

	_selectedCellIds = { cellId };
	_selectedParticleIds.clear();
}

void DataEditorModel::setParticle(ParticleDescription const & particle)
{
	_data.clear();
	_data.addParticle(particle);
	_unchangedData = _data;
	_navi.update(_data);

	_selectedCellIds.clear();
	_selectedParticleIds = { particle.id };
}

void DataEditorModel::setSelectionIds(set<uint64_t> const& selectedCellIds, set<uint64_t> const& selectedParticleIds)
{
	_data.clear();
	_unchangedData = _data;
	_navi.update(_data);

	_selectedCellIds = selectedCellIds;
	_selectedParticleIds = selectedParticleIds;
}

DataChangeDescription DataEditorModel::getAndUpdateChanges()
{
	DataChangeDescription result(_unchangedData, _data);
	_unchangedData = _data;
	return result;
}

CellDescription & DataEditorModel::getCellToEditRef()
{
	uint64_t selectedCellId = *_selectedCellIds.begin();
	int clusterIndex = _navi.clusterIndicesByCellIds.at(selectedCellId);
	int cellIndex = _navi.cellIndicesByCellIds.at(selectedCellId);
	return _data.clusters->at(clusterIndex).cells->at(cellIndex);
}

ParticleDescription & DataEditorModel::getParticleToEditRef()
{
	uint64_t selectedParticleId = *_selectedParticleIds.begin();
	int particleIndex = _navi.particleIndicesByParticleIds.at(selectedParticleId);
	return _data.particles->at(particleIndex);
}

ClusterDescription & DataEditorModel::getClusterToEditRef()
{
	uint64_t selectedCellId = *_selectedCellIds.begin();
	int clusterIndex = _navi.clusterIndicesByCellIds.at(selectedCellId);
	return _data.clusters->at(clusterIndex);
}

int DataEditorModel::getNumCells() const
{
	return _selectedCellIds.size();
}

int DataEditorModel::getNumParticles() const
{
	return _selectedParticleIds.size();
}

SimulationParameters * DataEditorModel::getSimulationParameters() const
{
	return _parameters;
}

void DataEditorModel::setSimulationParameters(SimulationParameters * parameters)
{
	_parameters = parameters;
}

map<string, string>& DataEditorModel::getSymbolsRef()
{
	return _symbols;
}
