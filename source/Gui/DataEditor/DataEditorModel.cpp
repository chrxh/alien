#include "Model/Api/ChangeDescriptions.h"

#include "DataEditorModel.h"

DataEditorModel::DataEditorModel(QObject *parent)
	: QObject(parent)
{

}

void DataEditorModel::editClusterAndCell(ClusterDescription const & cluster, uint64_t cellId)
{
	_data.clear();
	_data.addCluster(cluster);
	_unchangedData = _data;

	_selectedCellIds = { cellId };
	_selectedParticleIds.clear();
	_navi.update(_data);
}

DataChangeDescription DataEditorModel::getAndsUpdateChanges()
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

ClusterDescription & DataEditorModel::getClusterToEditRef()
{
	uint64_t selectedCellId = *_selectedCellIds.begin();
	int clusterIndex = _navi.clusterIndicesByCellIds.at(selectedCellId);
	return _data.clusters->at(clusterIndex);
}
