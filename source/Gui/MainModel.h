#pragma once
#include <QObject>

#include "Model/Api/Definitions.h"

class MainModel : public QObject {
	Q_OBJECT

public:
	MainModel(QObject * parent = nullptr);
	virtual ~MainModel() = default;

	SimulationParameters const* getSimulationParameters() const;
	void setSimulationParameters(SimulationParameters const* parameters);

	SymbolTable const* getSymbolTable() const;
	void setSymbolTable(SymbolTable const* symbols);

	optional<bool> isEditMode() const;
	void setEditMode(optional<bool> value);

private:
	SimulationParameters const* _parameters = nullptr;
	SymbolTable const* _symbols = nullptr;

	optional<bool> _isEditMode;
};
