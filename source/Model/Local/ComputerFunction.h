#pragma once

#include "CellFunction.h"

class ComputerFunction
	: public CellFunction
{
public:
    ComputerFunction (UnitContext* context) : CellFunction(context) {}
    virtual ~ComputerFunction () {}

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
