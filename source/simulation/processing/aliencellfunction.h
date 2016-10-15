#ifndef ALIENCELLFUNCTION_H
#define ALIENCELLFUNCTION_H

#include "../entities/alientoken.h"

class AlienCellCluster;
class AlienGrid;
class AlienEnergy;
class AlienCellFunction
{
public:
    AlienCellFunction(AlienGrid*& grid);
    virtual ~AlienCellFunction();

    virtual void runEnergyGuidanceSystem (AlienToken* token, AlienCell* cell, AlienCell* previousCell);
    virtual void execute (AlienToken* token, AlienCell* cell, AlienCell* previousCell, AlienEnergy*& newParticle, bool& decompose) = 0;
    virtual QString getCode ();
    virtual bool compileCode (QString code, int& errorLine);
    virtual QString getCellFunctionName () const = 0;

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

    qreal convertDataToAngle (quint8 b) const;
    quint8 convertAngleToData (qreal a) const;
    qreal convertDataToShiftLen (quint8 b) const;
    quint8 convertShiftLenToData (qreal len) const;
    quint8 convertURealToData (qreal r) const;
    qreal convertDataToUReal (quint8 d) const;
    quint8 convertIntToData (int i) const;
};

#endif // ALIENCELLFUNCTION_H
