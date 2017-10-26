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
	void editParticle(ParticleDescription const& particle);
	DataChangeDescription getAndsUpdateChanges();

	CellDescription& getCellToEditRef();
	ParticleDescription& getParticleToEditRef();
	ClusterDescription& getClusterToEditRef();

private:
	DataDescription _data;
	DataDescription _unchangedData;
	DescriptionNavigator _navi;

	set<uint64_t> _selectedCellIds;
	set<uint64_t> _selectedParticleIds;
};
