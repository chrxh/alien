#include "EngineInterfaceBuilderFacadeImpl.h"

#include "CellComputerCompilerImpl.h"
#include "DescriptionHelperImpl.h"
#include "EngineInterfaceSettings.h"
#include "SerializerImpl.h"
#include "SimulationChangerImpl.h"

DescriptionHelper* EngineInterfaceBuilderFacadeImpl::buildDescriptionHelper() const
{
    return new DescriptionHelperImpl();
}

CellComputerCompiler* EngineInterfaceBuilderFacadeImpl::buildCellComputerCompiler(
    SymbolTable* symbolTable,
    SimulationParameters const& parameters) const
{
    auto result = new CellComputerCompilerImpl();
    result->init(symbolTable, parameters);
    return result;
}

SimulationChanger* EngineInterfaceBuilderFacadeImpl::buildSimulationChanger(
    SimulationMonitor* monitor,
    NumberGenerator* numberGenerator) const
{
    auto result = new SimulationChangerImpl();
    result->init(monitor, numberGenerator);
    return result;
}

Serializer* EngineInterfaceBuilderFacadeImpl::buildSerializer() const
{
    return new SerializerImpl();
}

SymbolTable* EngineInterfaceBuilderFacadeImpl::getDefaultSymbolTable() const
{
    return EngineInterfaceSettings::getDefaultSymbolTable();
}

SimulationParameters EngineInterfaceBuilderFacadeImpl::getDefaultSimulationParameters() const
{
    return EngineInterfaceSettings::getDefaultSimulationParameters();
}

ExecutionParameters EngineInterfaceBuilderFacadeImpl::getDefaultExecutionParameters() const
{
    return EngineInterfaceSettings::getDefaultExecutionParameters();
}
