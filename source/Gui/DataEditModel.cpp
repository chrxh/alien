#include "DataEditModel.h"

#include "Base/DebugMacros.h"
#include "EngineInterface/ChangeDescriptions.h"

#include "Gui/DataRepository.h"

DataEditModel::DataEditModel(QObject *parent)
	: QObject(parent)
{
}

void DataEditModel::init(DataRepository* manipulator, SimulationParameters const& parameters, SymbolTable* symbols)
{
	_manipulator = manipulator;
	_parameters = parameters;
	_symbols = symbols;
}

void DataEditModel::setClusterAndCell(ClusterDescription const & cluster, uint64_t cellId)
{
    TRY;
	_data.clear();
	_data.addCluster(cluster);
	_unchangedData = _data;
	_navi.update(_data);

    if (_navi.cellIds.find(cellId) != _navi.cellIds.end()) {
	    _selectedCellIds = { cellId };
    }
	_selectedParticleIds.clear();
    CATCH;
}

void DataEditModel::setParticle(ParticleDescription const & particle)
{
    TRY;
    _data.clear();
	_data.addParticle(particle);
	_unchangedData = _data;
	_navi.update(_data);

	_selectedCellIds.clear();
    _selectedParticleIds = {particle.id};
    CATCH;
}

void DataEditModel::setSelectionIds(unordered_set<uint64_t> const& selectedCellIds, unordered_set<uint64_t> const& selectedParticleIds)
{
    TRY;
    _data.clear();
	_unchangedData = _data;
	_navi.update(_data);

	_selectedCellIds.clear();
	for (auto const& selectedCellId : selectedCellIds) {
        if (_navi.cellIds.find(selectedCellId) != _navi.cellIds.end()) {
            _selectedCellIds.insert(selectedCellId);
        }
    }

	_selectedParticleIds.clear();
    for (auto const& selectedParticleId : selectedParticleIds) {
        if (_navi.particleIds.find(selectedParticleId) != _navi.particleIds.end()) {
            _selectedParticleIds.insert(selectedParticleId);
        }
    }
    CATCH;
}

void DataEditModel::setSelectedTokenIndex(boost::optional<uint> const & value)
{
    TRY;
    _manipulator->setSelectedTokenIndex(value);
    CATCH;
}

boost::optional<uint> DataEditModel::getSelectedTokenIndex() const
{
    TRY;
    return _manipulator->getSelectedTokenIndex();
    CATCH;
}

DataChangeDescription DataEditModel::getAndUpdateChanges()
{
    TRY;
    DataChangeDescription result(_unchangedData, _data);
	_unchangedData = _data;
	return result;
    CATCH;
}

TokenDescription & DataEditModel::getTokenToEditRef(int tokenIndex)
{
    TRY;
    auto& cell = getCellToEditRef();
	return cell.tokens->at(tokenIndex);
    CATCH;
}

CellDescription & DataEditModel::getCellToEditRef()
{
    TRY;
    uint64_t selectedCellId = *_selectedCellIds.begin();
	int clusterIndex = _navi.clusterIndicesByCellIds.at(selectedCellId);
	int cellIndex = _navi.cellIndicesByCellIds.at(selectedCellId);
	return _data.clusters->at(clusterIndex).cells->at(cellIndex);
    CATCH;
}

ParticleDescription & DataEditModel::getParticleToEditRef()
{
    TRY;
    uint64_t selectedParticleId = *_selectedParticleIds.begin();
	int particleIndex = _navi.particleIndicesByParticleIds.at(selectedParticleId);
	return _data.particles->at(particleIndex);
    CATCH;
}

ClusterDescription & DataEditModel::getClusterToEditRef()
{
    TRY;
    uint64_t selectedCellId = *_selectedCellIds.begin();
	int clusterIndex = _navi.clusterIndicesByCellIds.at(selectedCellId);
	return _data.clusters->at(clusterIndex);
    CATCH;
}

int DataEditModel::getNumCells() const
{
    return _selectedCellIds.size();
}

int DataEditModel::getNumParticles() const
{
    return _selectedParticleIds.size();
}

SimulationParameters const& DataEditModel::getSimulationParameters() const
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

