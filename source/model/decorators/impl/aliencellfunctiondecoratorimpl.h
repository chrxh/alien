#ifndef ALIENCELLFUNCTIONDECORATORIMPL_H
#define ALIENCELLFUNCTIONDECORATORIMPL_H

#include "model/decorators/aliencellfunctiondecorator.h"

class AlienCellFunctionDecoratorImpl : AlienCellFunctionDecorator
{
public:
    AlienCellFunctionDecoratorImpl (AlienCell* cell, AlienGrid*& grid);
    virtual ~AlienCellFunctionDecoratorImpl () {}

    virtual QString decompileInstructionCode () const { return QString(); }
    virtual CompilationState injectAndCompileInstructionCode (QString code) { return CompilationState(); }

protected:
    AlienGrid*& _grid;
    qreal calcAngle (AlienCell* origin, AlienCell* ref1, AlienCell* ref2) const;

    static qreal convertDataToAngle (quint8 b);
    static quint8 convertAngleToData (qreal a);
    static qreal convertDataToShiftLen (quint8 b);
    static quint8 convertShiftLenToData (qreal len);
    static quint8 convertURealToData (qreal r);
    static qreal convertDataToUReal (quint8 d);
    static quint8 convertIntToData (int i);
};

#endif // ALIENCELLFUNCTIONDECORATORIMPL_H
