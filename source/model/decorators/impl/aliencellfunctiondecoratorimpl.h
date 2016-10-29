#ifndef ALIENCELLFUNCTIONDECORATOR_H
#define ALIENCELLFUNCTIONDECORATOR_H

#include "model/entities/alientoken.h"

class AlienCellCluster;
class AlienGrid;
class AlienEnergy;
class TestAlienCellFunctionCommunicator;
class AlienCellFunctionDecorator
{
public:
    AlienCellFunctionDecorator(AlienGrid*& grid);
    virtual ~AlienCellFunctionDecorator();

    virtual void runEnergyGuidanceSystem (AlienToken* token, AlienCell* cell, AlienCell* previousCell);
    virtual void execute (AlienToken* token, AlienCell* cell, AlienCell* previousCell, AlienEnergy*& newParticle, bool& decompose) = 0;
    virtual QString getCode ();
    virtual bool compileCode (QString code, int& errorLine);
    virtual AlienCellFunctionType getCellFunctionType () const = 0;

    virtual void serialize (QDataStream& stream);

    virtual void getInternalData (quint8* data);

    //constants for cell function programming
    enum class ENERGY_GUIDANCE {
        IN = 1,
        IN_VALUE_CELL = 2,
        IN_VALUE_TOKEN = 3
    };
    enum class ENERGY_GUIDANCE_IN {
        DEACTIVATED,
        BALANCE_CELL,
        BALANCE_TOKEN,
        BALANCE_BOTH,
        HARVEST_CELL,
        HARVEST_TOKEN
    };

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

#endif // ALIENCELLFUNCTIONDECORATOR_H
