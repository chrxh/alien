#pragma once

#include "ModelInterface/CellComputerCompiler.h"
#include "ModelInterface/Definitions.h"

class CellComputerCompilerLocal
	: public CellComputerCompiler
{
	Q_OBJECT
public:
	CellComputerCompilerLocal(QObject * parent = nullptr);
	virtual ~CellComputerCompilerLocal() = default;

	virtual void init(SymbolTable const* symbols, SimulationParameters const* parameters) = 0;
};
