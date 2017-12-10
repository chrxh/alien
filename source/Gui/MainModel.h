#pragma once
#include <QObject>

#include "Model/Api/Definitions.h"

class MainModel : public QObject {
	Q_OBJECT

public:
	MainModel(QObject * parent = nullptr);
	virtual ~MainModel() = default;

	SimulationParameters* getSimulationParameters() const;
	void setSimulationParameters(SimulationParameters* parameters);

	SymbolTable* getSymbolTable() const;
	void setSymbolTable(SymbolTable* symbols);

	optional<bool> isEditMode() const;
	void setEditMode(optional<bool> value);

private:
	SimulationParameters* _parameters = nullptr;
	SymbolTable* _symbols = nullptr;

	optional<bool> _isEditMode;
};
