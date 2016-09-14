#include "simulationparameters.h"
#include <QtGlobal>

SimulationParameters simulationParameters;

SimulationParameters::SimulationParameters ()
    : CRIT_CELL_DIST_MIN(0.3),
      CRIT_CELL_DIST_MAX(1.3),
      INTERNAL_TO_KINETIC_ENERGY(1),   //related to 1/mass
      CELL_MAX_FORCE(0.8),       //0.2
      CELL_MAX_FORCE_DECAY_PROB(0.2),
      MAX_CELL_CONNECTIONS(6),
      CELL_TOKENSTACKSIZE(9),
      MAX_TOKEN_ACCESS_NUMBERS(6), //1-16
      NEW_CELL_ENERGY(100.0),
      NEW_CELL_MAX_CONNECTION(4),
      NEW_CELL_TOKEN_ACCESS_NUMBER(0),
      CRIT_CELL_TRANSFORM_ENERGY(50.0),
      CELL_TRANSFORM_PROB(0.2),
      CLUSTER_FUSION_VEL(0.4),

      CELL_WEAPON_STRENGTH(0.1),
      CELL_CODESIZE(15),
      CELL_MEMSIZE(8),
      TOKEN_MEMSIZE(256),
      CELL_FUNCTION_CONSTRUCTOR_OFFSPRING_DIST(1.0),
      CELL_FUNCTION_SENSOR_RANGE(100),

      NEW_TOKEN_ENERGY(60.0),
      MIN_TOKEN_ENERGY(3.0),

      RAD_EXPONENT(1.0),
      RAD_FACTOR(0.0001),
      RAD_PROBABILITY(0.01),
      CELL_RAD_ENERGY_VEL_MULT(1.0),
      CELL_RAD_ENERGY_VEL_PERTURB(0.5)
{

}

void SimulationParameters::serializeData (QDataStream& stream)
{
    stream << CRIT_CELL_DIST_MIN;
    stream << CRIT_CELL_DIST_MAX;
    stream << INTERNAL_TO_KINETIC_ENERGY;
    stream << CELL_MAX_FORCE;
    stream << CELL_MAX_FORCE_DECAY_PROB;
    stream << MAX_CELL_CONNECTIONS;
    stream << CELL_TOKENSTACKSIZE;
    stream << MAX_TOKEN_ACCESS_NUMBERS;
    stream << NEW_CELL_ENERGY;
    stream << NEW_CELL_MAX_CONNECTION;
    stream << NEW_CELL_TOKEN_ACCESS_NUMBER;
    stream << CRIT_CELL_TRANSFORM_ENERGY;
    stream << CELL_TRANSFORM_PROB;
    stream << CLUSTER_FUSION_VEL;
    stream << CELL_WEAPON_STRENGTH;
    stream << CELL_CODESIZE;
    stream << CELL_MEMSIZE;
    stream << TOKEN_MEMSIZE;
    stream << CELL_FUNCTION_CONSTRUCTOR_OFFSPRING_DIST;
    stream << CELL_FUNCTION_SENSOR_RANGE;
    stream << NEW_TOKEN_ENERGY;
    stream << MIN_TOKEN_ENERGY;
    stream << RAD_EXPONENT;
    stream << RAD_FACTOR;
    stream << RAD_PROBABILITY;
    stream << CELL_RAD_ENERGY_VEL_MULT;
    stream << CELL_RAD_ENERGY_VEL_PERTURB;
}

void SimulationParameters::readData (QDataStream& stream)
{
    stream >> CRIT_CELL_DIST_MIN;
    stream >> CRIT_CELL_DIST_MAX;
    stream >> INTERNAL_TO_KINETIC_ENERGY;
    stream >> CELL_MAX_FORCE;
    stream >> CELL_MAX_FORCE_DECAY_PROB;
    stream >> MAX_CELL_CONNECTIONS;
    stream >> CELL_TOKENSTACKSIZE;
    stream >> MAX_TOKEN_ACCESS_NUMBERS;
    stream >> NEW_CELL_ENERGY;
    stream >> NEW_CELL_MAX_CONNECTION;
    stream >> NEW_CELL_TOKEN_ACCESS_NUMBER;
    stream >> CRIT_CELL_TRANSFORM_ENERGY;
    stream >> CELL_TRANSFORM_PROB;
    stream >> CLUSTER_FUSION_VEL;
    stream >> CELL_WEAPON_STRENGTH;
    stream >> CELL_CODESIZE;
    stream >> CELL_MEMSIZE;
    stream >> TOKEN_MEMSIZE;
    stream >> CELL_FUNCTION_CONSTRUCTOR_OFFSPRING_DIST;
    stream >> CELL_FUNCTION_SENSOR_RANGE;
    stream >> NEW_TOKEN_ENERGY;
    stream >> MIN_TOKEN_ENERGY;
    stream >> RAD_EXPONENT;
    stream >> RAD_FACTOR;
    stream >> RAD_PROBABILITY;
    stream >> CELL_RAD_ENERGY_VEL_MULT;
    stream >> CELL_RAD_ENERGY_VEL_PERTURB;
}
