#ifndef ALIENCELLFUNCTIONSENSOR_H
#define ALIENCELLFUNCTIONSENSOR_H

#include "aliencellfunction.h"

class AlienCellFunctionSensor : public AlienCellFunction
{
public:
    AlienCellFunctionSensor ();
    AlienCellFunctionSensor (quint8* cellTypeData);
    AlienCellFunctionSensor (QDataStream& stream);

    void execute (AlienToken* token, AlienCell* previousCell, AlienCell* cell, AlienGrid* grid, AlienEnergy*& newParticle, bool& decompose);
    QString getCellFunctionName ();

    void serialize (QDataStream& stream);

    //constants for cell function programming
    enum class SENSOR {
        OUT = 5,
        IN = 20,
        INOUT_ANGLE = 21,
        IN_MIN_MASS = 22,
        IN_MAX_MASS = 23,
        OUT_MASS = 24,
        OUT_DIST = 25
    };
    enum class SENSOR_IN {
        DO_NOTHING,
        SEARCH_VICINITY,
        SEARCH_BY_ANGLE,
        SEARCH_FROM_CENTER,
        SEARCH_TOWARD_CENTER
    };
    enum class SENSOR_OUT {
        NOTHING_FOUND,
        CLUSTER_FOUND
    };
};

#endif // ALIENCELLFUNCTIONSENSOR_H
