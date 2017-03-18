#ifndef CONFIG_H
#define CONFIG_H

#include "definitions.h"

const qreal ALIEN_PRECISION = 0.0000001;

class Metadata
{
public:
    static void loadDefaultSymbolTable (SymbolTable* meta);
	static void loadDefaultSimulationParameters(SimulationParameters* parameters);
};

/*struct SimulationParameters
{
    //simulation constants
    qreal CRIT_CELL_DIST_MIN;
    qreal CRIT_CELL_DIST_MAX;
    qreal INTERNAL_TO_KINETIC_ENERGY; //related to 1/mass
    qreal CELL_MAX_FORCE;
    qreal CELL_MAX_FORCE_DECAY_PROB;
    int MAX_CELL_CONNECTIONS;
    int CELL_TOKENSTACKSIZE;
    int MAX_TOKEN_ACCESS_NUMBERS;
    qreal NEW_CELL_ENERGY;
    int NEW_CELL_MAX_CONNECTION;
    int NEW_CELL_TOKEN_ACCESS_NUMBER;
    qreal CRIT_CELL_TRANSFORM_ENERGY;
    qreal CELL_TRANSFORM_PROB;
    qreal CLUSTER_FUSION_VEL;

    qreal CELL_WEAPON_STRENGTH;
    int CELL_NUM_INSTR;
    int CELL_MEMSIZE;
    int TOKEN_MEMSIZE;
    qreal CELL_FUNCTION_CONSTRUCTOR_OFFSPRING_DIST;
    qreal CELL_FUNCTION_SENSOR_RANGE;
    qreal CELL_FUNCTION_COMMUNICATOR_RANGE;

    qreal NEW_TOKEN_ENERGY;
    qreal MIN_TOKEN_ENERGY;

    qreal RAD_EXPONENT;
    qreal RAD_FACTOR;
    qreal RAD_PROBABILITY;

    qreal CELL_RAD_ENERGY_VEL_MULT;
    qreal CELL_RAD_ENERGY_VEL_PERTURB;

    SimulationParameters ();

    void serializeData (QDataStream& stream);
    void deserializeData (QDataStream& stream);
};

extern SimulationParameters simulationParameters;
*/
#endif // CONFIG_H
