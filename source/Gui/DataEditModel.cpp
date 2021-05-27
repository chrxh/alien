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

    _selectedCellIds = selectedCellIds;
    _selectedParticleIds = selectedParticleIds;

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

boost::optional<TokenDescription&> DataEditModel::getTokenToEditRef(int tokenIndex)
{
    TRY;
    if (auto cell = getCellToEditRef()) {
        return cell->tokens->at(tokenIndex);
    }
    return boost::none;
    CATCH;
}

boost::optional<CellDescription&> DataEditModel::getCellToEditRef()
{
    TRY;
    if (_selectedCellIds.empty()) {
        return boost::none;
    }
    uint64_t selectedCellId = *_selectedCellIds.begin();

    if (_navi.clusterIndicesByCellIds.find(selectedCellId) == _navi.clusterIndicesByCellIds.end()) {
        return boost::none;
    }
    if (_navi.cellIndicesByCellIds.find(selectedCellId) == _navi.cellIndicesByCellIds.end()) {
        return boost::none;
    }

	int clusterIndex = _navi.clusterIndicesByCellIds.at(selectedCellId);
	int cellIndex = _navi.cellIndicesByCellIds.at(selectedCellId);
	return _data.clusters->at(clusterIndex).cells->at(cellIndex);
    CATCH;
}

boost::optional<ParticleDescription&> DataEditModel::getParticleToEditRef()
{
    TRY;
    if (_selectedParticleIds.empty()) {
        return boost::none;
    }

    uint64_t selectedParticleId = *_selectedParticleIds.begin();
    if (_navi.particleIndicesByParticleIds.find(selectedParticleId) == _navi.particleIndicesByParticleIds.end()) {
        return boost::none;
    }

    int particleIndex = _navi.particleIndicesByParticleIds.at(selectedParticleId);
	return _data.particles->at(particleIndex);
    CATCH;
}

boost::optional<ClusterDescription&> DataEditModel::getClusterToEditRef()
{
    TRY;
    if (_selectedCellIds.empty()) {
        return boost::none;
    }
    uint64_t selectedCellId = *_selectedCellIds.begin();
    if (_navi.clusterIndicesByCellIds.find(selectedCellId) == _navi.clusterIndicesByCellIds.end()) {
        return boost::none;
    }

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

