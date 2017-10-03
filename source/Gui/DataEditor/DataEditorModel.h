#pragma once

#include <QObject>

#include "Model/Api/Descriptions.h"

class DataEditorModel
	: public QObject
{
	Q_OBJECT

public:
	DataEditorModel(QObject *parent);
	virtual ~DataEditorModel() = default;

	void editClusterAndCell(ClusterDescription const& cluster, uint64_t cellId);
	DataChangeDescription getAndsUpdateChanges();

	CellDescription& getCellToEditRef();
	ClusterDescription& getClusterToEditRef();

private:
	DataDescription _data;
	DataDescription _unchangedData;
	DescriptionNavigationMaps _navi;

	set<uint64_t> _selectedCellIds;
	set<uint64_t> _selectedParticleIds;
};
