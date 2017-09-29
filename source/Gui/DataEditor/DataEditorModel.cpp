#include "DataEditorModel.h"

DataEditorModel::DataEditorModel(QObject *parent)
	: QObject(parent)
{

}

void DataEditorModel::editClusterAndCell(ClusterDescription const & cluster, uint64_t cellId)
{
	_selectedData.clear();
	_selectedData.addCluster(cluster);
	_selectedCellIds = { cellId };
	_selectedParticleIds.clear();
	_navi.update(_selectedData);
}

CellDescription & DataEditorModel::getCellToEditRef()
{
	uint64_t selectedCellId = *_selectedCellIds.begin();
	int clusterIndex = _navi.clusterIndicesByCellIds.at(selectedCellId);
	int cellIndex = _navi.cellIndicesByCellIds.at(selectedCellId);
	return _selectedData.clusters->at(clusterIndex).cells->at(cellIndex);
}

ClusterDescription & DataEditorModel::getClusterToEditRef()
{
	uint64_t selectedCellId = *_selectedCellIds.begin();
	int clusterIndex = _navi.clusterIndicesByCellIds.at(selectedCellId);
	return _selectedData.clusters->at(clusterIndex);
}
