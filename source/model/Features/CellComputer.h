#ifndef CELLFUNCTIONCOMPUTER_H
#define CELLFUNCTIONCOMPUTER_H

#include "CellFunction.h"

class CellComputer
	: public CellFunction
{
public:
    CellComputer (UnitContext* context) : CellFunction(context) {}
    virtual ~CellComputer () {}

    Enums::CellFunction::Type getType () const { return Enums::CellFunction::COMPUTER; }

    //new interface
    virtual QString decompileInstructionCode () const = 0;
    struct CompilationState {
        bool compilationOk;
        int errorAtLine;
    };
    virtual CompilationState injectAndCompileInstructionCode (QString code) = 0;
    virtual QByteArray& getMemoryReference () = 0;
};

#endif // CELLFUNCTIONCOMPUTER_H
