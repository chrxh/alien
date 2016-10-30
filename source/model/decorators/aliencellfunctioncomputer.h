#ifndef ALIENCELLFUNCTIONCOMPUTER_H
#define ALIENCELLFUNCTIONCOMPUTER_H

#include "aliencellfunction.h"

class AlienCellFunctionComputer: public AlienCellFunction
{
public:
    AlienCellFunctionComputer (AlienCell* cell) : AlienCellFunction(cell) {}
    virtual ~AlienCellFunctionComputer () {}

    virtual QString decompileInstructionCode () const = 0;
    virtual CompilationState injectAndCompileInstructionCode (QString code) = 0;

    CellFunctionType getType () const { return CellFunctionType::COMPUTER; }
    virtual void getInternalData (quint8* data) const = 0;

    struct CompilationState {
        bool compilationOk = true;
        int errorAtLine = 0;
    };

};

#endif // ALIENCELLFUNCTIONCOMPUTER_H
