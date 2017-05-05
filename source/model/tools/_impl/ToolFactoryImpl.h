#ifndef TOOLFACTORYIMPL_H
#define TOOLFACTORYIMPL_H

#include "model/tools/ToolFactory.h"

class ToolFactoryImpl
	: public ToolFactory
{
public:
	ToolFactoryImpl();
	virtual ~ToolFactoryImpl() = default;

	virtual SimulationAccess* buildSimulationAccess() const override;
};

#endif // TOOLFACTORYIMPL_H
