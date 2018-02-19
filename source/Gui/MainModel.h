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

	SymbolTable* getSymbolTable() const;
	void setSymbolTable(SymbolTable* symbols);

	int getTPS() const;
	void setTPS(int value);

private:
	SimulationParameters const* _parameters = nullptr;
	SymbolTable* _symbols = nullptr;

	int _tps = 50;
};
