#pragma once

#include <QColor>

#include "Definitions.h"

namespace Const
{
	const QColor IndividualCellColor1(0x50, 0x70, 0xFF, 0xB0);
	const QColor IndividualCellColor2(0xFF, 0x60, 0x40, 0xB0);
	const QColor IndividualCellColor3(0x70, 0xFF, 0x50, 0xB0);
	const QColor IndividualCellColor4(0xFF, 0xBF, 0x50, 0xB0);
	const QColor IndividualCellColor5(0xBF, 0x50, 0xFF, 0xB0);
	const QColor IndividualCellColor6(0x50, 0xFF, 0xEF, 0xB0);
	const QColor IndividualCellColor7(0xBF, 0xBF, 0xBF, 0xB0);
	const QColor CellFunctionInfoColor(0x40, 0x40, 0x90);
	const QColor BranchNumberInfoColor(0x00, 0x00, 0x00);
}

class MODELBASIC_EXPORT ModelSettings
{
public:
    static SymbolTable* getDefaultSymbolTable();
	static SimulationParameters getDefaultSimulationParameters();
    static ExecutionParameters getDefaultExecutionParameters();
};

