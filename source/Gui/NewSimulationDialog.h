#pragma once

#include <QDialog>

#include "ModelBasic/Definitions.h"
#include "ModelBasic/SimulationParameters.h"

#include "Definitions.h"

namespace Ui {
	class NewSimulationDialog;
}

class SimulationParametersDialog;
class SymbolTableDialog;
class NewSimulationDialog : public QDialog
{
	Q_OBJECT

public:
	NewSimulationDialog(SimulationParameters const& parameters, SymbolTable const* symbols, Serializer* serializer, QWidget* parent = nullptr);
	virtual ~NewSimulationDialog();

	SimulationConfig getConfig() const;
	double getEnergy() const;

private:
	IntVector2D getUniverseSizeForModelCpu() const;
	IntVector2D getUniverseSizeForModelGpu() const;
	IntVector2D getGridSize() const;
	uint getMaxThreads() const;
	SymbolTable* getSymbolTable() const;
	SimulationParameters const& getSimulationParameters() const;
	IntVector2D getUnitSize() const;
    ModelComputationType getModelType() const;
    uint getNumBlocks() const;
    uint getNumThreadsPerBlock() const;
    uint getMaxClusters() const;
    uint getMaxCells() const;
    uint getMaxTokens() const;
    uint getMaxParticles() const;
    uint getDynamicMemorySize() const;

private:
	Q_SLOT void simulationParametersButtonClicked();
	Q_SLOT void symbolTableButtonClicked();
	Q_SLOT void updateLabels();
	Q_SLOT void okClicked();

private:
	Ui::NewSimulationDialog *ui;
	Serializer* _serializer = nullptr;

	SimulationParameters _parameters;
	SymbolTable* _symbolTable = nullptr;

	IntVector2D _universeSizeForModelCpu;
	IntVector2D _universeSizeForModelGpu;
	IntVector2D _gridSize;
};

