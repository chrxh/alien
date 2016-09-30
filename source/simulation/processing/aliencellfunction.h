#ifndef ALIENCELLFUNCTION_H
#define ALIENCELLFUNCTION_H

#include "../entities/alientoken.h"

class AlienCellCluster;
class AlienGrid;
class AlienEnergy;
class AlienCellFunction
{
public:
    AlienCellFunction();
    virtual ~AlienCellFunction();

    virtual void runEnergyGuidanceSystem (AlienToken* token, AlienCell* previousCell, AlienCell* cell, AlienGrid*& space);
    virtual void execute (AlienToken* token, AlienCell* previousCell, AlienCell* cell, AlienGrid*& space, AlienEnergy*& newParticle, bool& decompose) = 0;
    virtual QString getCode ();
    virtual bool compileCode (QString code, int& errorLine);
    virtual QString getCellFunctionName () = 0;

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
    qreal convertDataToAngle (quint8 b);
    quint8 convertAngleToData (qreal a);
    qreal convertDataToShiftLen (quint8 b);
    quint8 convertShiftLenToData (qreal len);
    quint8 convertURealToData (qreal r);
};

#endif // ALIENCELLFUNCTION_H
