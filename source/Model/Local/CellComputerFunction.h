#pragma once

#include "CellFunction.h"

class CellComputerFunction
	: public CellFunction
{
public:
    CellComputerFunction (UnitContext* context) : CellFunction(context) {}
    virtual ~CellComputerFunction () {}

    Enums::CellFunction::Type getType () const { return Enums::CellFunction::COMPUTER; }

    virtual QString decompileInstructionCode () const = 0;
    struct CompilationState {
        bool compilationOk;
        int errorAtLine;
    };
    virtual CompilationState injectAndCompileInstructionCode (QString code) = 0;
};
