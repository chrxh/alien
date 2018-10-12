#pragma once

#include <QObject>

#include "ModelBasic/Descriptions.h"
#include "Gui/Definitions.h"

class DataEditModel
	: public QObject
{
	Q_OBJECT

public:
	DataEditModel(QObject *parent);
	virtual ~DataEditModel() = default;

	void init(DataRepository* manipulator, SimulationParameters const* parameters, SymbolTable* symbols);

	void setClusterAndCell(ClusterDescription const& cluster, uint64_t cellId);
	void setParticle(ParticleDescription const& particle);
	void setSelectionIds(unordered_set<uint64_t> const& selectedCellIds, unordered_set<uint64_t> const& selectedParticleIds);
	void setSelectedTokenIndex(optional<uint> const& value);	//will be stored in DataManipulator
	optional<uint> getSelectedTokenIndex() const;

	DataChangeDescription getAndUpdateChanges();

	TokenDescription& getTokenToEditRef(int tokenIndex);
	CellDescription& getCellToEditRef();
	ParticleDescription& getParticleToEditRef();
	ClusterDescription& getClusterToEditRef();

	int getNumCells() const;
	int getNumParticles() const;

	SimulationParameters const* getSimulationParameters() const;
	SymbolTable* getSymbolTable() const;
	void setSymbolTable(SymbolTable* symbols);

private:
	DataDescription _data;
	DataDescription _unchangedData;
	DescriptionNavigator _navi;

	unordered_set<uint64_t> _selectedCellIds;
	unordered_set<uint64_t> _selectedParticleIds;

	DataRepository* _manipulator = nullptr;
	SimulationParameters const* _parameters = nullptr;
	SymbolTable* _symbols = nullptr;
};
