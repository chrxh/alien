#include "ModelBasicBuilderFacadeImpl.h"

#include "CellComputerCompilerImpl.h"
#include "DescriptionHelperImpl.h"
#include "SerializerImpl.h"
#include "ModelBasicSettings.h"

DescriptionHelper * ModelBasicBuilderFacadeImpl::buildDescriptionHelper() const
{
	return new DescriptionHelperImpl();
}

CellComputerCompiler * ModelBasicBuilderFacadeImpl::buildCellComputerCompiler(SymbolTable* symbolTable, SimulationParameters const& parameters) const
{
	auto result = new CellComputerCompilerImpl();
	result->init(symbolTable, parameters);
	return result;
}

Serializer * ModelBasicBuilderFacadeImpl::buildSerializer() const
{
	return new SerializerImpl();
}

SymbolTable * ModelBasicBuilderFacadeImpl::getDefaultSymbolTable() const
{
	return ModelBasicSettings::getDefaultSymbolTable();
}

SimulationParameters ModelBasicBuilderFacadeImpl::getDefaultSimulationParameters() const
{
	return ModelBasicSettings::getDefaultSimulationParameters();
}

ExecutionParameters ModelBasicBuilderFacadeImpl::getDefaultExecutionParameters() const
{
    return ModelBasicSettings::getDefaultExecutionParameters();
}


