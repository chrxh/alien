#ifndef ALIENCELLFUNCTIONSENSOR_H
#define ALIENCELLFUNCTIONSENSOR_H

#include "aliencellfunction.h"

class AlienCellFunctionSensor : public AlienCellFunction
{
public:
    AlienCellFunctionSensor (AlienGrid*& grid);
    AlienCellFunctionSensor (quint8* cellFunctionData, AlienGrid*& grid);
    AlienCellFunctionSensor (QDataStream& stream, AlienGrid*& grid);

    void execute (AlienToken* token, AlienCell* cell, AlienCell* previousCell, AlienEnergy*& newParticle, bool& decompose);

    void serialize (QDataStream& stream);

    //constants for cell function programming
    enum class SENSOR {
        OUT = 5,
        IN = 20,
        INOUT_ANGLE = 21,
        IN_MIN_MASS = 22,
        IN_MAX_MASS = 23,
        OUT_MASS = 24,
        OUT_DISTANCE = 25
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
