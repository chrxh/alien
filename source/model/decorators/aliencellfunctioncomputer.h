#ifndef ALIENCELLFUNCTIONCOMPUTER_H
#define ALIENCELLFUNCTIONCOMPUTER_H

#include "aliencellfunction.h"

class AlienCellFunctionComputer: public AlienCellFunction
{
public:
    AlienCellFunctionComputer (AlienGrid*& grid) : AlienCellFunction(grid) {}
    virtual ~AlienCellFunctionComputer () {}

    CellFunctionType getType () const { return CellFunctionType::COMPUTER; }

    //new interface
    virtual QString decompileInstructionCode () const = 0;
    struct CompilationState {
        bool compilationOk;
        int errorAtLine;
    };
    virtual CompilationState injectAndCompileInstructionCode (QString code) = 0;
    virtual QVector< quint8 >& getMemoryReference () = 0;
};

#endif // ALIENCELLFUNCTIONCOMPUTER_H
