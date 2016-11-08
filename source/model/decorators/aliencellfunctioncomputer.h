#ifndef ALIENCELLFUNCTIONCOMPUTER_H
#define ALIENCELLFUNCTIONCOMPUTER_H

#include "aliencellfunction.h"

class AlienCellFunctionComputer: public AlienCellFunction
{
public:
    AlienCellFunctionComputer (AlienCell* cell, AlienGrid*& grid) : AlienCellFunction(cell, grid) {}
    virtual ~AlienCellFunctionComputer () {}

    virtual QString decompileInstructionCode () const = 0;
    struct CompilationState {
        bool compilationOk;
        int errorAtLine;
    };
    virtual CompilationState injectAndCompileInstructionCode (QString code) = 0;
    virtual QVector< quint8 >& getMemoryReference () = 0;

    CellFunctionType getType () const { return CellFunctionType::COMPUTER; }
};

#endif // ALIENCELLFUNCTIONCOMPUTER_H
