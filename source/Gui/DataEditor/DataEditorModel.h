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

	void setClusterAndCell(ClusterDescription const& cluster, uint64_t cellId);
	void setParticle(ParticleDescription const& particle);
	void setSelectionIds(unordered_set<uint64_t> const& selectedCellIds, unordered_set<uint64_t> const& selectedParticleIds);

	DataChangeDescription getAndUpdateChanges();

	CellDescription& getCellToEditRef();
	ParticleDescription& getParticleToEditRef();
	ClusterDescription& getClusterToEditRef();

	int getNumCells() const;
	int getNumParticles() const;

	SimulationParameters* getSimulationParameters() const;
	void setSimulationParameters(SimulationParameters* parameters);

	map<string, string>& getSymbolsRef();

private:
	DataDescription _data;
	DataDescription _unchangedData;
	DescriptionNavigator _navi;

	unordered_set<uint64_t> _selectedCellIds;
	unordered_set<uint64_t> _selectedParticleIds;

	SimulationParameters* _parameters = nullptr;
	map<string, string> _symbols;
};
