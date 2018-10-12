#include "ModelBasicBuilderFacadeImpl.h"

#include "DescriptionHelperImpl.h"
#include "SerializerImpl.h"
#include "Settings.h"

DescriptionHelper * ModelBasicBuilderFacadeImpl::buildDescriptionHelper() const
{
	return new DescriptionHelperImpl();
}

Serializer * ModelBasicBuilderFacadeImpl::buildSerializer() const
{
	return new SerializerImpl();
}

SymbolTable * ModelBasicBuilderFacadeImpl::buildDefaultSymbolTable() const
{
	return ModelSettings::getDefaultSymbolTable();
}

SimulationParameters* ModelBasicBuilderFacadeImpl::buildDefaultSimulationParameters() const
{
	return ModelSettings::getDefaultSimulationParameters();
}


