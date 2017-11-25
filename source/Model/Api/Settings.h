#pragma once

#include <QColor>

#include "Definitions.h"

const qreal ALIEN_PRECISION = 0.0000001;

const QColor INDIVIDUAL_CELL_COLOR1(0x50, 0x90, 0xFF, 0xB0);
const QColor INDIVIDUAL_CELL_COLOR2(0xFF, 0x60, 0x40, 0xB0);
const QColor INDIVIDUAL_CELL_COLOR3(0x70, 0xFF, 0x50, 0xB0);
const QColor INDIVIDUAL_CELL_COLOR4(0xFF, 0xBF, 0x50, 0xB0);
const QColor INDIVIDUAL_CELL_COLOR5(0xBF, 0x50, 0xFF, 0xB0);
const QColor INDIVIDUAL_CELL_COLOR6(0x50, 0xFF, 0xEF, 0xB0);
const QColor INDIVIDUAL_CELL_COLOR7(0xBF, 0xBF, 0xBF, 0xB0);
const QColor CELLFUNCTION_INFO_COLOR(0x40, 0x40, 0x90);
const QColor BRANCHNUMBER_INFO_COLOR(0x00, 0x00, 0x00);

class MODEL_EXPORT ModelSettings
{
public:
    static SymbolTable* loadDefaultSymbolTable();
	static SimulationParameters* loadDefaultSimulationParameters();
};

