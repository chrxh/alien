#pragma once

#include "Model/Api/Definitions.h"
#include "Model/Api/CellComputerCompiler.h"

class CellComputerCompilerImpl
	: public CellComputerCompiler
{
	Q_OBJECT
public:
	CellComputerCompilerImpl(QObject * parent = nullptr);
	virtual ~CellComputerCompilerImpl() = default;

	void init(SymbolTable* symbols, SimulationParameters* parameters);

	virtual CompilationResult compileSourceCode(std::string const& code) const override;
	virtual std::string decompileSourceCode(QByteArray const& data) const override;

private:
	SymbolTable* _symbols = nullptr;
	SimulationParameters* _parameters = nullptr;
};
