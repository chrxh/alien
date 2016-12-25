#ifndef CELLFUNCTIONCOMPUTER_H
#define CELLFUNCTIONCOMPUTER_H

#include "cellfunction.h"

class CellFunctionComputer: public CellFunction
{
public:
    CellFunctionComputer (SimulationContext* context) : CellFunction(context) {}
    virtual ~CellFunctionComputer () {}

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

#endif // CELLFUNCTIONCOMPUTER_H
