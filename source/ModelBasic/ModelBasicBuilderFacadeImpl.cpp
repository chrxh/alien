#include "ModelBasicBuilderFacadeImpl.h"

#include "CellComputerCompilerImpl.h"
#include "DescriptionHelperImpl.h"
#include "SerializerImpl.h"
#include "Settings.h"

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

SymbolTable * ModelBasicBuilderFacadeImpl::buildDefaultSymbolTable() const
{
	return ModelSettings::getDefaultSymbolTable();
}

SimulationParameters ModelBasicBuilderFacadeImpl::buildDefaultSimulationParameters() const
{
	return ModelSettings::getDefaultSimulationParameters();
}


