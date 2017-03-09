#include <QtGlobal>

#include "model/features/cellfeatureconstants.h"
#include "model/metadata/symboltable.h"

#include "config.h"

void Metadata::loadDefaultSymbolTable(SymbolTable* symbolTable)
{
    symbolTable->clearTable();

    //general variables
    symbolTable->addEntry("i","[255]");
    symbolTable->addEntry("j","[254]");
    symbolTable->addEntry("k","[253]");
    symbolTable->addEntry("l","[252]");

    //token branch number
    symbolTable->addEntry("BRANCH_NUMBER","[0]");

    //energy guidance system
    symbolTable->addEntry("ENERGY_GUIDANCE_IN","["+QString::number(static_cast<int>(ENERGY_GUIDANCE::IN))+"]");
    symbolTable->addEntry("ENERGY_GUIDANCE_IN::DEACTIVATED",QString::number(static_cast<int>(ENERGY_GUIDANCE_IN::DEACTIVATED)));
    symbolTable->addEntry("ENERGY_GUIDANCE_IN::BALANCE_CELL",QString::number(static_cast<int>(ENERGY_GUIDANCE_IN::BALANCE_CELL)));
    symbolTable->addEntry("ENERGY_GUIDANCE_IN::BALANCE_TOKEN",QString::number(static_cast<int>(ENERGY_GUIDANCE_IN::BALANCE_TOKEN)));
    symbolTable->addEntry("ENERGY_GUIDANCE_IN::BALANCE_BOTH",QString::number(static_cast<int>(ENERGY_GUIDANCE_IN::BALANCE_BOTH)));
    symbolTable->addEntry("ENERGY_GUIDANCE_IN::HARVEST_CELL",QString::number(static_cast<int>(ENERGY_GUIDANCE_IN::HARVEST_CELL)));
    symbolTable->addEntry("ENERGY_GUIDANCE_IN::HARVEST_TOKEN",QString::number(static_cast<int>(ENERGY_GUIDANCE_IN::HARVEST_TOKEN)));
    symbolTable->addEntry("ENERGY_GUIDANCE_IN_VALUE_CELL","["+QString::number(static_cast<int>(ENERGY_GUIDANCE::IN_VALUE_CELL))+"]");
    symbolTable->addEntry("ENERGY_GUIDANCE_IN_VALUE_TOKEN","["+QString::number(static_cast<int>(ENERGY_GUIDANCE::IN_VALUE_TOKEN))+"]");

    //constructor
    symbolTable->addEntry("CONSTR_OUT","["+QString::number(static_cast<int>(CONSTR::OUT))+"]");
    symbolTable->addEntry("CONSTR_OUT::SUCCESS",QString::number(static_cast<int>(CONSTR_OUT::SUCCESS)));
    symbolTable->addEntry("CONSTR_OUT::SUCCESS_ROT",QString::number(static_cast<int>(CONSTR_OUT::SUCCESS_ROT)));
    symbolTable->addEntry("CONSTR_OUT::ERROR_NO_ENERGY",QString::number(static_cast<int>(CONSTR_OUT::ERROR_NO_ENERGY)));
    symbolTable->addEntry("CONSTR_OUT::ERROR_OBSTACLE",QString::number(static_cast<int>(CONSTR_OUT::ERROR_OBSTACLE)));
    symbolTable->addEntry("CONSTR_OUT::ERROR_CONNECTION",QString::number(static_cast<int>(CONSTR_OUT::ERROR_CONNECTION)));
    symbolTable->addEntry("CONSTR_OUT::ERROR_DIST",QString::number(static_cast<int>(CONSTR_OUT::ERROR_DIST)));
    symbolTable->addEntry("CONSTR_IN","["+QString::number(static_cast<int>(CONSTR::IN))+"]");
    symbolTable->addEntry("CONSTR_IN::DO_NOTHING",QString::number(static_cast<int>(CONSTR_IN::DO_NOTHING)));
    symbolTable->addEntry("CONSTR_IN::SAFE",QString::number(static_cast<int>(CONSTR_IN::SAFE)));
    symbolTable->addEntry("CONSTR_IN::UNSAFE",QString::number(static_cast<int>(CONSTR_IN::UNSAFE)));
    symbolTable->addEntry("CONSTR_IN::BRUTEFORCE",QString::number(static_cast<int>(CONSTR_IN::BRUTEFORCE)));
    symbolTable->addEntry("CONSTR_IN_OPTION","["+QString::number(static_cast<int>(CONSTR::IN_OPTION))+"]");
    symbolTable->addEntry("CONSTR_IN_OPTION::STANDARD",QString::number(static_cast<int>(CONSTR_IN_OPTION::STANDARD)));
    symbolTable->addEntry("CONSTR_IN_OPTION::CREATE_EMPTY_TOKEN",QString::number(static_cast<int>(CONSTR_IN_OPTION::CREATE_EMPTY_TOKEN)));
    symbolTable->addEntry("CONSTR_IN_OPTION::CREATE_DUP_TOKEN",QString::number(static_cast<int>(CONSTR_IN_OPTION::CREATE_DUP_TOKEN)));
    symbolTable->addEntry("CONSTR_IN_OPTION::FINISH_NO_SEP",QString::number(static_cast<int>(CONSTR_IN_OPTION::FINISH_NO_SEP)));
    symbolTable->addEntry("CONSTR_IN_OPTION::FINISH_WITH_SEP",QString::number(static_cast<int>(CONSTR_IN_OPTION::FINISH_WITH_SEP)));
    symbolTable->addEntry("CONSTR_IN_OPTION::FINISH_WITH_SEP_RED",QString::number(static_cast<int>(CONSTR_IN_OPTION::FINISH_WITH_SEP_RED)));
    symbolTable->addEntry("CONSTR_IN_OPTION::FINISH_WITH_TOKEN_SEP_RED",QString::number(static_cast<int>(CONSTR_IN_OPTION::FINISH_WITH_TOKEN_SEP_RED)));
    symbolTable->addEntry("CONSTR_INOUT_ANGLE","["+QString::number(static_cast<int>(CONSTR::INOUT_ANGLE))+"]");
    symbolTable->addEntry("CONSTR_IN_DIST","["+QString::number(static_cast<int>(CONSTR::IN_DIST))+"]");
    symbolTable->addEntry("CONSTR_IN_CELL_MAX_CONNECTIONS","["+QString::number(static_cast<int>(CONSTR::IN_CELL_MAX_CONNECTIONS))+"]");
    symbolTable->addEntry("CONSTR_IN_CELL_MAX_CONNECTIONS::AUTO","0");       //artificial entry (has no symbol in enum class)
    symbolTable->addEntry("CONSTR_IN_CELL_BRANCH_NO","["+QString::number(static_cast<int>(CONSTR::IN_CELL_BRANCH_NO))+"]");
    symbolTable->addEntry("CONSTR_IN_CELL_FUNCTION","["+QString::number(static_cast<int>(CONSTR::IN_CELL_FUNCTION))+"]");
    symbolTable->addEntry("CONSTR_IN_CELL_FUNCTION::COMPUTER",QString::number(static_cast<int>(CellFunctionType::COMPUTER)));
    symbolTable->addEntry("CONSTR_IN_CELL_FUNCTION::PROP",QString::number(static_cast<int>(CellFunctionType::PROPULSION)));
    symbolTable->addEntry("CONSTR_IN_CELL_FUNCTION::SCANNER",QString::number(static_cast<int>(CellFunctionType::SCANNER)));
    symbolTable->addEntry("CONSTR_IN_CELL_FUNCTION::WEAPON",QString::number(static_cast<int>(CellFunctionType::WEAPON)));
    symbolTable->addEntry("CONSTR_IN_CELL_FUNCTION::CONSTR",QString::number(static_cast<int>(CellFunctionType::CONSTRUCTOR)));
    symbolTable->addEntry("CONSTR_IN_CELL_FUNCTION::SENSOR",QString::number(static_cast<int>(CellFunctionType::SENSOR)));
    symbolTable->addEntry("CONSTR_IN_CELL_FUNCTION::COMMUNICATOR",QString::number(static_cast<int>(CellFunctionType::COMMUNICATOR)));
    symbolTable->addEntry("CONSTR_IN_CELL_FUNCTION_DATA","["+QString::number(static_cast<int>(CONSTR::IN_CELL_FUNCTION_DATA))+"]");

    //propulsion
    symbolTable->addEntry("PROP_OUT","["+QString::number(static_cast<int>(PROP::OUT))+"]");
    symbolTable->addEntry("PROP_OUT::SUCCESS",QString::number(static_cast<int>(PROP_OUT::SUCCESS)));
    symbolTable->addEntry("PROP_OUT::SUCCESS_DAMPING_FINISHED",QString::number(static_cast<int>(PROP_OUT::SUCCESS_DAMPING_FINISHED)));
    symbolTable->addEntry("PROP_OUT::ERROR_NO_ENERGY",QString::number(static_cast<int>(PROP_OUT::ERROR_NO_ENERGY)));
    symbolTable->addEntry("PROP_IN","["+QString::number(static_cast<int>(PROP::IN))+"]");
    symbolTable->addEntry("PROP_IN::DO_NOTHING",QString::number(static_cast<int>(PROP_IN::DO_NOTHING)));
    symbolTable->addEntry("PROP_IN::BY_ANGLE",QString::number(static_cast<int>(PROP_IN::BY_ANGLE)));
    symbolTable->addEntry("PROP_IN::FROM_CENTER",QString::number(static_cast<int>(PROP_IN::FROM_CENTER)));
    symbolTable->addEntry("PROP_IN::TOWARD_CENTER",QString::number(static_cast<int>(PROP_IN::TOWARD_CENTER)));
    symbolTable->addEntry("PROP_IN::ROTATION_CLOCKWISE",QString::number(static_cast<int>(PROP_IN::ROTATION_CLOCKWISE)));
    symbolTable->addEntry("PROP_IN::ROTATION_COUNTERCLOCKWISE",QString::number(static_cast<int>(PROP_IN::ROTATION_COUNTERCLOCKWISE)));
    symbolTable->addEntry("PROP_IN::DAMP_ROTATION",QString::number(static_cast<int>(PROP_IN::DAMP_ROTATION)));
    symbolTable->addEntry("PROP_IN_ANGLE","["+QString::number(static_cast<int>(PROP::IN_ANGLE))+"]");
    symbolTable->addEntry("PROP_IN_POWER","["+QString::number(static_cast<int>(PROP::IN_POWER))+"]");

    //scanner
    symbolTable->addEntry("SCANNER_OUT","["+QString::number(static_cast<int>(SCANNER::OUT))+"]");
    symbolTable->addEntry("SCANNER_OUT::SUCCESS",QString::number(static_cast<int>(SCANNER_OUT::SUCCESS)));
    symbolTable->addEntry("SCANNER_OUT::FINISHED",QString::number(static_cast<int>(SCANNER_OUT::FINISHED)));
    symbolTable->addEntry("SCANNER_OUT::RESTART",QString::number(static_cast<int>(SCANNER_OUT::RESTART)));
//    meta->addEntry("SCANNER_IN","[11]");
    symbolTable->addEntry("SCANNER_INOUT_CELL_NUMBER","["+QString::number(static_cast<int>(SCANNER::INOUT_CELL_NUMBER))+"]");
    symbolTable->addEntry("SCANNER_OUT_MASS","["+QString::number(static_cast<int>(SCANNER::OUT_MASS))+"]");
    symbolTable->addEntry("SCANNER_OUT_ENERGY","["+QString::number(static_cast<int>(SCANNER::OUT_ENERGY))+"]");
    symbolTable->addEntry("SCANNER_OUT_ANGLE","["+QString::number(static_cast<int>(SCANNER::OUT_ANGLE))+"]");
    symbolTable->addEntry("SCANNER_OUT_DISTANCE","["+QString::number(static_cast<int>(SCANNER::OUT_DISTANCE))+"]");
    symbolTable->addEntry("SCANNER_OUT_CELL_MAX_CONNECTIONS","["+QString::number(static_cast<int>(SCANNER::OUT_CELL_MAX_CONNECTIONS))+"]");
    symbolTable->addEntry("SCANNER_OUT_CELL_BRANCH_NO","["+QString::number(static_cast<int>(SCANNER::OUT_CELL_BRANCH_NO))+"]");
    symbolTable->addEntry("SCANNER_OUT_CELL_FUNCTION","["+QString::number(static_cast<int>(SCANNER::OUT_CELL_FUNCTION))+"]");
    symbolTable->addEntry("SCANNER_OUT_CELL_FUNCTION::COMPUTER",QString::number(static_cast<int>(CellFunctionType::COMPUTER)));
    symbolTable->addEntry("SCANNER_OUT_CELL_FUNCTION::PROP",QString::number(static_cast<int>(CellFunctionType::PROPULSION)));
    symbolTable->addEntry("SCANNER_OUT_CELL_FUNCTION::SCANNER",QString::number(static_cast<int>(CellFunctionType::SCANNER)));
    symbolTable->addEntry("SCANNER_OUT_CELL_FUNCTION::WEAPON",QString::number(static_cast<int>(CellFunctionType::WEAPON)));
    symbolTable->addEntry("SCANNER_OUT_CELL_FUNCTION::CONSTR",QString::number(static_cast<int>(CellFunctionType::CONSTRUCTOR)));
    symbolTable->addEntry("SCANNER_OUT_CELL_FUNCTION::SENSOR",QString::number(static_cast<int>(CellFunctionType::SENSOR)));
    symbolTable->addEntry("SCANNER_OUT_CELL_FUNCTION::COMMUNICATOR",QString::number(static_cast<int>(CellFunctionType::COMMUNICATOR)));
    symbolTable->addEntry("SCANNER_OUT_CELL_FUNCTION_DATA","["+QString::number(static_cast<int>(SCANNER::OUT_CELL_FUNCTION_DATA))+"]");

    //weapon
    symbolTable->addEntry("WEAPON_OUT","["+QString::number(static_cast<int>(WEAPON::OUT))+"]");
    symbolTable->addEntry("WEAPON_OUT::NO_TARGET",QString::number(static_cast<int>(WEAPON_OUT::NO_TARGET)));
    symbolTable->addEntry("WEAPON_OUT::STRIKE_SUCCESSFUL",QString::number(static_cast<int>(WEAPON_OUT::STRIKE_SUCCESSFUL)));

    //sensor
    symbolTable->addEntry("SENSOR_OUT", "["+QString::number(static_cast<int>(SENSOR::OUT))+"]");
    symbolTable->addEntry("SENSOR_OUT::NOTHING_FOUND", QString::number(static_cast<int>(SENSOR_OUT::NOTHING_FOUND)));
    symbolTable->addEntry("SENSOR_OUT::CLUSTER_FOUND", QString::number(static_cast<int>(SENSOR_OUT::CLUSTER_FOUND)));
    symbolTable->addEntry("SENSOR_IN", "["+QString::number(static_cast<int>(SENSOR::IN))+"]");
    symbolTable->addEntry("SENSOR_IN::DO_NOTHING", QString::number(static_cast<int>(SENSOR_IN::DO_NOTHING)));
    symbolTable->addEntry("SENSOR_IN::SEARCH_VICINITY", QString::number(static_cast<int>(SENSOR_IN::SEARCH_VICINITY)));
    symbolTable->addEntry("SENSOR_IN::SEARCH_BY_ANGLE", QString::number(static_cast<int>(SENSOR_IN::SEARCH_BY_ANGLE)));
    symbolTable->addEntry("SENSOR_IN::SEARCH_FROM_CENTER", QString::number(static_cast<int>(SENSOR_IN::SEARCH_FROM_CENTER)));
    symbolTable->addEntry("SENSOR_IN::SEARCH_TOWARD_CENTER", QString::number(static_cast<int>(SENSOR_IN::SEARCH_TOWARD_CENTER)));
    symbolTable->addEntry("SENSOR_INOUT_ANGLE", "["+QString::number(static_cast<int>(SENSOR::INOUT_ANGLE))+"]");
    symbolTable->addEntry("SENSOR_IN_MIN_MASS", "["+QString::number(static_cast<int>(SENSOR::IN_MIN_MASS))+"]");
    symbolTable->addEntry("SENSOR_IN_MAX_MASS", "["+QString::number(static_cast<int>(SENSOR::IN_MAX_MASS))+"]");
    symbolTable->addEntry("SENSOR_OUT_MASS", "["+QString::number(static_cast<int>(SENSOR::OUT_MASS))+"]");
    symbolTable->addEntry("SENSOR_OUT_DISTANCE", "["+QString::number(static_cast<int>(SENSOR::OUT_DISTANCE))+"]");

    //communicator
    symbolTable->addEntry("COMMUNICATOR_IN", "["+QString::number(static_cast<int>(COMMUNICATOR::IN))+"]");
    symbolTable->addEntry("COMMUNICATOR_IN::DO_NOTHING", QString::number(static_cast<int>(COMMUNICATOR_IN::DO_NOTHING)));
    symbolTable->addEntry("COMMUNICATOR_IN::SET_LISTENING_CHANNEL", QString::number(static_cast<int>(COMMUNICATOR_IN::SET_LISTENING_CHANNEL)));
    symbolTable->addEntry("COMMUNICATOR_IN::SEND_MESSAGE", QString::number(static_cast<int>(COMMUNICATOR_IN::SEND_MESSAGE)));
    symbolTable->addEntry("COMMUNICATOR_IN::RECEIVE_MESSAGE", QString::number(static_cast<int>(COMMUNICATOR_IN::RECEIVE_MESSAGE)));
    symbolTable->addEntry("COMMUNICATOR_IN_CHANNEL", "["+QString::number(static_cast<int>(COMMUNICATOR::IN_CHANNEL))+"]");
    symbolTable->addEntry("COMMUNICATOR_IN_MESSAGE", "["+QString::number(static_cast<int>(COMMUNICATOR::IN_MESSAGE))+"]");
    symbolTable->addEntry("COMMUNICATOR_IN_ANGLE", "["+QString::number(static_cast<int>(COMMUNICATOR::IN_ANGLE))+"]");
    symbolTable->addEntry("COMMUNICATOR_IN_DISTANCE", "["+QString::number(static_cast<int>(COMMUNICATOR::IN_DISTANCE))+"]");
    symbolTable->addEntry("COMMUNICATOR_OUT_SENT_NUM_MESSAGE", "["+QString::number(static_cast<int>(COMMUNICATOR::OUT_SENT_NUM_MESSAGE))+"]");
    symbolTable->addEntry("COMMUNICATOR_OUT_RECEIVED_NEW_MESSAGE", "["+QString::number(static_cast<int>(COMMUNICATOR::OUT_RECEIVED_NEW_MESSAGE))+"]");
    symbolTable->addEntry("COMMUNICATOR_OUT_RECEIVED_NEW_MESSAGE::NO", QString::number(static_cast<int>(COMMUNICATOR_OUT_RECEIVED_NEW_MESSAGE::NO)));
    symbolTable->addEntry("COMMUNICATOR_OUT_RECEIVED_NEW_MESSAGE::YES", QString::number(static_cast<int>(COMMUNICATOR_OUT_RECEIVED_NEW_MESSAGE::YES)));
    symbolTable->addEntry("COMMUNICATOR_OUT_RECEIVED_MESSAGE", "["+QString::number(static_cast<int>(COMMUNICATOR::OUT_RECEIVED_MESSAGE))+"]");
    symbolTable->addEntry("COMMUNICATOR_OUT_RECEIVED_ANGLE", "["+QString::number(static_cast<int>(COMMUNICATOR::OUT_RECEIVED_ANGLE))+"]");
    symbolTable->addEntry("COMMUNICATOR_OUT_RECEIVED_DISTANCE", "["+QString::number(static_cast<int>(COMMUNICATOR::OUT_RECEIVED_DISTANCE))+"]");
}

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
      CELL_NUM_INSTR(15),
      CELL_MEMSIZE(8),
      TOKEN_MEMSIZE(256),
      CELL_FUNCTION_CONSTRUCTOR_OFFSPRING_DIST(1.0),
      CELL_FUNCTION_SENSOR_RANGE(100.0),
      CELL_FUNCTION_COMMUNICATOR_RANGE(30.0),

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
    stream << CELL_NUM_INSTR;
    stream << CELL_MEMSIZE;
    stream << TOKEN_MEMSIZE;
    stream << CELL_FUNCTION_CONSTRUCTOR_OFFSPRING_DIST;
    stream << CELL_FUNCTION_SENSOR_RANGE;
    stream << CELL_FUNCTION_COMMUNICATOR_RANGE;
    stream << NEW_TOKEN_ENERGY;
    stream << MIN_TOKEN_ENERGY;
    stream << RAD_EXPONENT;
    stream << RAD_FACTOR;
    stream << RAD_PROBABILITY;
    stream << CELL_RAD_ENERGY_VEL_MULT;
    stream << CELL_RAD_ENERGY_VEL_PERTURB;
}

void SimulationParameters::deserializeData (QDataStream& stream)
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
    stream >> CELL_NUM_INSTR;
    stream >> CELL_MEMSIZE;
    stream >> TOKEN_MEMSIZE;
    stream >> CELL_FUNCTION_CONSTRUCTOR_OFFSPRING_DIST;
    stream >> CELL_FUNCTION_SENSOR_RANGE;
    stream >> CELL_FUNCTION_COMMUNICATOR_RANGE;
    stream >> NEW_TOKEN_ENERGY;
    stream >> MIN_TOKEN_ENERGY;
    stream >> RAD_EXPONENT;
    stream >> RAD_FACTOR;
    stream >> RAD_PROBABILITY;
    stream >> CELL_RAD_ENERGY_VEL_MULT;
    stream >> CELL_RAD_ENERGY_VEL_PERTURB;
}
