#pragma once
#include <QObject>

#include "ModelBasic/Definitions.h"

class MainModel : public QObject {
	Q_OBJECT

public:
	MainModel(QObject * parent = nullptr);
	virtual ~MainModel() = default;

	SimulationParameters getSimulationParameters() const;
	void setSimulationParameters(SimulationParameters const& parameters);

    ExecutionParameters getExecutionParameters() const;
    void setExecutionParameters(ExecutionParameters const& parameters);

	SymbolTable* getSymbolTable() const;
	void setSymbolTable(SymbolTable* symbols);

	int getTPS() const;
	void setTPS(int value);

private:
	SimulationParameters _simulationParameters;
    ExecutionParameters _executionParameters;
    SymbolTable* _symbols = nullptr;

	int _tps = 50;
};
