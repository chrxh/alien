#ifndef ALIENCELLFUNCTIONCONSTRUCTOR_H
#define ALIENCELLFUNCTIONCONSTRUCTOR_H

#include "aliencellfunction.h"

#include <QVector3D>

class AlienCellCluster;
class AlienCellFunctionConstructor : public AlienCellFunction
{
public:
    AlienCellFunctionConstructor (AlienGrid*& grid);
    AlienCellFunctionConstructor (quint8* cellFunctionData, AlienGrid*& grid);
    AlienCellFunctionConstructor (QDataStream& stream, AlienGrid*& grid);

    void execute (AlienToken* token, AlienCell* cell, AlienCell* previousCell, AlienEnergy*& newParticle, bool& decompose);
    QString getCellFunctionName () const;

    void serialize (QDataStream& stream);

    //constants for cell function programming
    enum class CONSTR {
        OUT = 5,
        IN = 6,
        IN_OPTION = 7,
        INOUT_ANGLE = 15,
        IN_DIST = 16,
        IN_CELL_MAX_CONNECTIONS = 17,              //0: automatically; >0: max connections (not greater than MAX_CELL_CONNECTIONS)
        IN_CELL_BRANCH_NO = 18,
        IN_CELL_FUNCTION = 19,
        IN_CELL_FUNCTION_DATA = 40
    };
    enum class CONSTR_OUT {
        SUCCESS,
        SUCCESS_ROT,
        ERROR_NO_ENERGY,
        ERROR_OBSTACLE,
        ERROR_CONNECTION,
        ERROR_DIST
    };
    enum class CONSTR_IN {
        DO_NOTHING,
        SAFE,
        UNSAFE,
        BRUTEFORCE
    };
    enum class CONSTR_IN_OPTION {
        STANDARD,
        CREATE_EMPTY_TOKEN,
        CREATE_DUP_TOKEN,
        FINISH_NO_SEP,
        FINISH_WITH_SEP,
        FINISH_WITH_SEP_RED,
        FINISH_WITH_TOKEN_SEP_RED
    };
    enum class CONSTR_IN_CELL_FUNCTION {
        COMPUTER,
        PROP,
        SCANNER,
        WEAPON,
        CONSTR,
        SENSOR,
        COMMUNICATOR
    };

private:
    AlienCell* constructNewCell (AlienCell* baseCell,
                                 QVector3D posOfNewCell,
                                 int maxConnections,
                                 int tokenAccessNumber,
                                 int cellType,
                                 quint8* cellFunctionData);
    AlienCell* obstacleCheck (AlienCellCluster* cluster, bool safeMode);
    qreal averageEnergy (qreal e1, qreal e2);
    void separateConstruction (AlienCell* constructedCell,
                               AlienCell* constructorCell,
                               bool reduceConnection);
    QString convertCellTypeNumberToName (int type);

};

#endif // ALIENCELLFUNCTIONCONSTRUCTOR_H
