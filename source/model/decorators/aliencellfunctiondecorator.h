#ifndef ALIENCELLFUNCTIONDECORATOR_H
#define ALIENCELLFUNCTIONDECORATOR_H

#include "model/entities/aliencell.h"

class AlienCellFunctionDecorator : public AlienCell
{
public:
    AlienCellFunctionDecorator (AlienCell* cell) : _cell(cell) {}
    virtual ~AlienCellFunctionDecorator () {}

    virtual QString decompileInstructionCode () const = 0;
    virtual CompilationState injectAndCompileInstructionCode (QString code) = 0;
    virtual void getInternalData (quint8* data) const = 0;

    struct CompilationState {
        bool compilationOk = true;
        int errorAtLine = 0;
    };

private:
    AlienCell* _cell;
};

#endif // ALIENCELLFUNCTIONDECORATOR_H

