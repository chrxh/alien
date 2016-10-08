#ifndef ALIENTOKENPROCESSINGSCANNER_H
#define ALIENTOKENPROCESSINGSCANNER_H

#include "aliencellfunction.h"

class AlienCellFunctionScanner : public AlienCellFunction
{
public:
    AlienCellFunctionScanner ();
    AlienCellFunctionScanner (quint8* cellTypeData);
    AlienCellFunctionScanner (QDataStream& stream);

    void execute (AlienToken* token, AlienCell* cell, AlienCell* previousCell, AlienGrid* grid, AlienEnergy*& newParticle, bool& decompose);
    QString getCellFunctionName () const;

    void serialize (QDataStream& stream);

    //constants for cell function programming
    enum class SCANNER {
        OUT = 5,
        INOUT_CELL_NUMBER = 12,
        OUT_MASS = 13,
        OUT_ENERGY = 14,
        OUT_ANGLE = 15,
        OUT_DIST = 16,
        OUT_CELL_MAX_CONNECTIONS = 17,
        OUT_CELL_BRANCH_NO = 18,
        OUT_CELL_FUNCTION = 19,
        OUT_CELL_FUNCTION_DATA = 35
    };
    enum class SCANNER_OUT {
        SUCCESS,
        FINISHED,
        RESTART
    };
    enum class SCANNER_OUT_CELL_FUNCTION {
        COMPUTER,
        PROP,
        SCANNER,
        WEAPON,
        CONSTR,
        SENSOR,
        COMMUNICATOR
    };

protected:
    void spiralLookupAlgorithm (AlienCell*& cell, AlienCell*& previousCell1, AlienCell*& previousCell2, int n, const quint64& tag);
    int convertCellTypeNameToNumber(QString type);
};

#endif // ALIENTOKENPROCESSINGSCANNER_H
