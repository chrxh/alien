#pragma once

#include "ModelInterface/Definitions.h"
#include "ModelInterface/CellComputerCompiler.h"

class CellComputerCompilerImpl
	: public CellComputerCompiler
{
	Q_OBJECT
public:
	CellComputerCompilerImpl(QObject * parent = nullptr);
	virtual ~CellComputerCompilerImpl() = default;

	virtual void init(SymbolTable const* symbols, SimulationParameters const* parameters);

	virtual CompilationResult compileSourceCode(std::string const& code) const override;
	virtual std::string decompileSourceCode(QByteArray const& data) const override;

private:
	SymbolTable const* _symbols = nullptr;
	SimulationParameters const* _parameters = nullptr;
};
