#ifndef ALIENTOKENPROCESSINGSCANNER_H
#define ALIENTOKENPROCESSINGSCANNER_H

#include "aliencellfunction.h"

class AlienCellFunctionScanner : public AlienCellFunction
{
public:
    AlienCellFunctionScanner ();
    AlienCellFunctionScanner (quint8* cellTypeData);
    AlienCellFunctionScanner (QDataStream& stream);

    void execute (AlienToken* token, AlienCell* previousCell, AlienCell* cell, AlienGrid*& space, AlienEnergy*& newParticle, bool& decompose);
    QString getCellFunctionName ();

    void serialize (QDataStream& stream);

protected:
    void spiralLookupAlgorithm (AlienCell*& cell, AlienCell*& previousCell1, AlienCell*& previousCell2, int n, const quint64& tag);
    int convertCellTypeNameToNumber(QString type);
};

#endif // ALIENTOKENPROCESSINGSCANNER_H
