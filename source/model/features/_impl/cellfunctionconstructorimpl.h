#ifndef CELLFUNCTIONCONSTRUCTOR_H
#define CELLFUNCTIONCONSTRUCTOR_H

#include "model/features/cellfunction.h"

#include <QVector3D>

class CellFunctionConstructorImpl : public CellFunction
{
public:
    CellFunctionConstructorImpl (SimulationContext* context);

    Enums::CellFunction::Type getType () const { return Enums::CellFunction::CONSTRUCTOR; }

protected:
    ProcessingResult processImpl (Token* token, Cell* cell, Cell* previousCell) override;

private:
    CellMap* _cellMap = nullptr;
    Topology* _topology = nullptr;
	SimulationParameters* _parameters = nullptr;
};

#endif // CELLFUNCTIONCONSTRUCTOR_H
