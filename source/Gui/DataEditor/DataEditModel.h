#pragma once

#include <QObject>

#include "Model/Api/Descriptions.h"

class DataEditModel
	: public QObject
{
	Q_OBJECT

public:
	DataEditModel(QObject *parent);
	virtual ~DataEditModel() = default;

	void init(SimulationParameters const* parameters, SymbolTable* symbols);

	void setClusterAndCell(ClusterDescription const& cluster, uint64_t cellId);
	void setParticle(ParticleDescription const& particle);
	void setSelectionIds(unordered_set<uint64_t> const& selectedCellIds, unordered_set<uint64_t> const& selectedParticleIds);
	void setSelectedTokenIndex(uint value);

	uint getSelectedTokenIndex() const;

	DataChangeDescription getAndUpdateChanges();

	TokenDescription& getTokenToEditRef(int tokenIndex);
	CellDescription& getCellToEditRef();
	ParticleDescription& getParticleToEditRef();
	ClusterDescription& getClusterToEditRef();

	int getNumCells() const;
	int getNumParticles() const;

	SimulationParameters const* getSimulationParameters() const;
	SymbolTable* getSymbolTable() const;

private:
	DataDescription _data;
	DataDescription _unchangedData;
	DescriptionNavigator _navi;

	uint _selectedTokenIndex = 0;
	unordered_set<uint64_t> _selectedCellIds;
	unordered_set<uint64_t> _selectedParticleIds;

	SimulationParameters const* _parameters = nullptr;
	SymbolTable* _symbols = nullptr;
};
