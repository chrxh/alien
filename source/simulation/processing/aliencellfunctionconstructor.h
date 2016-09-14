#ifndef ALIENCELLFUNCTIONCONSTRUCTOR_H
#define ALIENCELLFUNCTIONCONSTRUCTOR_H

#include "aliencellfunction.h"

#include <QVector3D>

class AlienCellCluster;
class AlienCellFunctionConstructor : public AlienCellFunction
{
public:
    AlienCellFunctionConstructor ();
    AlienCellFunctionConstructor (quint8* cellTypeData);
    AlienCellFunctionConstructor (QDataStream& stream);

    void execute (AlienToken* token, AlienCell* previousCell, AlienCell* cell, AlienGrid*& grid, AlienEnergy*& newParticle, bool& decompose);
    QString getCellFunctionName ();

    void serialize (QDataStream& stream);

private:
    AlienCell* constructNewCell (AlienCell* baseCell, QVector3D posOfNewCell, int maxConnections, int tokenAccessNumber, int cellType, quint8* cellTypeData, AlienGrid*& grid);
    AlienCell* obstacleCheck (AlienCellCluster* cluster, bool safeMode, AlienGrid*& grid);
    qreal averageEnergy (qreal e1, qreal e2);
    void separateConstruction (AlienCell* constructedCell, AlienCell* constructorCell, bool reduceConnection);
    QString convertCellTypeNumberToName (int type);

};

#endif // ALIENCELLFUNCTIONCONSTRUCTOR_H
