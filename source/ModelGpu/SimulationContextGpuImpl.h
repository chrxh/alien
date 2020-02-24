#pragma once

#include <QThread>

#include "ModelBasic/SimulationContext.h"
#include "DefinitionsImpl.h"
#include "ModelGpuData.h"

class SimulationContextGpuImpl
	: public SimulationContext
{
	Q_OBJECT
public:
	SimulationContextGpuImpl(QObject* parent = nullptr);
	virtual ~SimulationContextGpuImpl() = default;

    void init(
        SpaceProperties* metric,
        int timestep,
        SymbolTable* symbolTable,
        SimulationParameters const& parameters,
        ModelGpuData const& specificData);

    virtual SpaceProperties* getSpaceProperties() const override;
	virtual SymbolTable* getSymbolTable() const override;
	virtual SimulationParameters const& getSimulationParameters() const override;
	virtual NumberGenerator* getNumberGenerator() const override;

	virtual map<string, int> getSpecificData() const override;
    virtual int getTimestep() const override;

	virtual void setSimulationParameters(SimulationParameters const& parameters) override;
    virtual void setExecutionParameters(ExecutionParameters const& parameters) override;

	virtual CudaController* getCudaController() const;

private:
	SpaceProperties *_metric = nullptr;
	SymbolTable *_symbolTable = nullptr;
	SimulationParameters _parameters;
	CudaController *_cudaController = nullptr;
	NumberGenerator* _numberGen = nullptr;
    ModelGpuData _specificData;
};
