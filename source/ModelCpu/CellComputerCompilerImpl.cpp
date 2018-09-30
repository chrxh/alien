#include "CellComputerCompilerImpl.h"

#include "CellComputerFunctionImpl.h"

CellComputerCompilerImpl::CellComputerCompilerImpl(QObject * parent) : CellComputerCompilerLocal(parent)
{
	
}

void CellComputerCompilerImpl::init(SymbolTable const* symbols, SimulationParameters const* parameters)
{
	_symbols = symbols;
	_parameters = parameters;
}

CompilationResult CellComputerCompilerImpl::compileSourceCode(std::string const & code) const
{
	return CellComputerFunctionImpl::compileSourceCode(code, _symbols);
}

std::string CellComputerCompilerImpl::decompileSourceCode(QByteArray const & data) const
{
	return CellComputerFunctionImpl::decompileSourceCode(data, _parameters);
}
