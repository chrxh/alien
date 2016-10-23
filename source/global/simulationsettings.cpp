#include "simulationsettings.h"
#include "model/metadatamanager.h"
#include "model/processing/aliencellfunction.h"
#include "model/processing/aliencellfunctionconstructor.h"
#include "model/processing/aliencellfunctionpropulsion.h"
#include "model/processing/aliencellfunctionscanner.h"
#include "model/processing/aliencellfunctionweapon.h"
#include "model/processing/aliencellfunctionsensor.h"
#include "model/processing/aliencellfunctioncommunicator.h"


#include <QtGlobal>

void AlienMetadata::loadDefaultMetadata (MetadataManager* meta)
{
    meta->clearSymbolTable();

    //general variables
    meta->addSymbolEntry("i","[255]");
    meta->addSymbolEntry("j","[254]");
    meta->addSymbolEntry("k","[253]");
    meta->addSymbolEntry("l","[252]");

    //token branch number
    meta->addSymbolEntry("BRANCH_NUMBER","[0]");

    //energy guidance system
    meta->addSymbolEntry("ENERGY_GUIDANCE_IN","["+QString::number(static_cast<int>(AlienCellFunction::ENERGY_GUIDANCE::IN))+"]");
    meta->addSymbolEntry("ENERGY_GUIDANCE_IN::DEACTIVATED",QString::number(static_cast<int>(AlienCellFunction::ENERGY_GUIDANCE_IN::DEACTIVATED)));
    meta->addSymbolEntry("ENERGY_GUIDANCE_IN::BALANCE_CELL",QString::number(static_cast<int>(AlienCellFunction::ENERGY_GUIDANCE_IN::BALANCE_CELL)));
    meta->addSymbolEntry("ENERGY_GUIDANCE_IN::BALANCE_TOKEN",QString::number(static_cast<int>(AlienCellFunction::ENERGY_GUIDANCE_IN::BALANCE_TOKEN)));
    meta->addSymbolEntry("ENERGY_GUIDANCE_IN::BALANCE_BOTH",QString::number(static_cast<int>(AlienCellFunction::ENERGY_GUIDANCE_IN::BALANCE_BOTH)));
    meta->addSymbolEntry("ENERGY_GUIDANCE_IN::HARVEST_CELL",QString::number(static_cast<int>(AlienCellFunction::ENERGY_GUIDANCE_IN::HARVEST_CELL)));
    meta->addSymbolEntry("ENERGY_GUIDANCE_IN::HARVEST_TOKEN",QString::number(static_cast<int>(AlienCellFunction::ENERGY_GUIDANCE_IN::HARVEST_TOKEN)));
    meta->addSymbolEntry("ENERGY_GUIDANCE_IN_VALUE_CELL","["+QString::number(static_cast<int>(AlienCellFunction::ENERGY_GUIDANCE::IN_VALUE_CELL))+"]");
    meta->addSymbolEntry("ENERGY_GUIDANCE_IN_VALUE_TOKEN","["+QString::number(static_cast<int>(AlienCellFunction::ENERGY_GUIDANCE::IN_VALUE_TOKEN))+"]");

    //constructor
    meta->addSymbolEntry("CONSTR_OUT","["+QString::number(static_cast<int>(AlienCellFunctionConstructor::CONSTR::OUT))+"]");
    meta->addSymbolEntry("CONSTR_OUT::SUCCESS",QString::number(static_cast<int>(AlienCellFunctionConstructor::CONSTR_OUT::SUCCESS)));
    meta->addSymbolEntry("CONSTR_OUT::SUCCESS_ROT",QString::number(static_cast<int>(AlienCellFunctionConstructor::CONSTR_OUT::SUCCESS_ROT)));
    meta->addSymbolEntry("CONSTR_OUT::ERROR_NO_ENERGY",QString::number(static_cast<int>(AlienCellFunctionConstructor::CONSTR_OUT::ERROR_NO_ENERGY)));
    meta->addSymbolEntry("CONSTR_OUT::ERROR_OBSTACLE",QString::number(static_cast<int>(AlienCellFunctionConstructor::CONSTR_OUT::ERROR_OBSTACLE)));
    meta->addSymbolEntry("CONSTR_OUT::ERROR_CONNECTION",QString::number(static_cast<int>(AlienCellFunctionConstructor::CONSTR_OUT::ERROR_CONNECTION)));
    meta->addSymbolEntry("CONSTR_OUT::ERROR_DIST",QString::number(static_cast<int>(AlienCellFunctionConstructor::CONSTR_OUT::ERROR_DIST)));
    meta->addSymbolEntry("CONSTR_IN","["+QString::number(static_cast<int>(AlienCellFunctionConstructor::CONSTR::IN))+"]");
    meta->addSymbolEntry("CONSTR_IN::DO_NOTHING",QString::number(static_cast<int>(AlienCellFunctionConstructor::CONSTR_IN::DO_NOTHING)));
    meta->addSymbolEntry("CONSTR_IN::SAFE",QString::number(static_cast<int>(AlienCellFunctionConstructor::CONSTR_IN::SAFE)));
    meta->addSymbolEntry("CONSTR_IN::UNSAFE",QString::number(static_cast<int>(AlienCellFunctionConstructor::CONSTR_IN::UNSAFE)));
    meta->addSymbolEntry("CONSTR_IN::BRUTEFORCE",QString::number(static_cast<int>(AlienCellFunctionConstructor::CONSTR_IN::BRUTEFORCE)));
    meta->addSymbolEntry("CONSTR_IN_OPTION","["+QString::number(static_cast<int>(AlienCellFunctionConstructor::CONSTR::IN_OPTION))+"]");
    meta->addSymbolEntry("CONSTR_IN_OPTION::STANDARD",QString::number(static_cast<int>(AlienCellFunctionConstructor::CONSTR_IN_OPTION::STANDARD)));
    meta->addSymbolEntry("CONSTR_IN_OPTION::CREATE_EMPTY_TOKEN",QString::number(static_cast<int>(AlienCellFunctionConstructor::CONSTR_IN_OPTION::CREATE_EMPTY_TOKEN)));
    meta->addSymbolEntry("CONSTR_IN_OPTION::CREATE_DUP_TOKEN",QString::number(static_cast<int>(AlienCellFunctionConstructor::CONSTR_IN_OPTION::CREATE_DUP_TOKEN)));
    meta->addSymbolEntry("CONSTR_IN_OPTION::FINISH_NO_SEP",QString::number(static_cast<int>(AlienCellFunctionConstructor::CONSTR_IN_OPTION::FINISH_NO_SEP)));
    meta->addSymbolEntry("CONSTR_IN_OPTION::FINISH_WITH_SEP",QString::number(static_cast<int>(AlienCellFunctionConstructor::CONSTR_IN_OPTION::FINISH_WITH_SEP)));
    meta->addSymbolEntry("CONSTR_IN_OPTION::FINISH_WITH_SEP_RED",QString::number(static_cast<int>(AlienCellFunctionConstructor::CONSTR_IN_OPTION::FINISH_WITH_SEP_RED)));
    meta->addSymbolEntry("CONSTR_IN_OPTION::FINISH_WITH_TOKEN_SEP_RED",QString::number(static_cast<int>(AlienCellFunctionConstructor::CONSTR_IN_OPTION::FINISH_WITH_TOKEN_SEP_RED)));
    meta->addSymbolEntry("CONSTR_INOUT_ANGLE","["+QString::number(static_cast<int>(AlienCellFunctionConstructor::CONSTR::INOUT_ANGLE))+"]");
    meta->addSymbolEntry("CONSTR_IN_DIST","["+QString::number(static_cast<int>(AlienCellFunctionConstructor::CONSTR::IN_DIST))+"]");
    meta->addSymbolEntry("CONSTR_IN_CELL_MAX_CONNECTIONS","["+QString::number(static_cast<int>(AlienCellFunctionConstructor::CONSTR::IN_CELL_MAX_CONNECTIONS))+"]");
    meta->addSymbolEntry("CONSTR_IN_CELL_MAX_CONNECTIONS::AUTO","0");       //artificial entry (has no symbol in enum class)
    meta->addSymbolEntry("CONSTR_IN_CELL_BRANCH_NO","["+QString::number(static_cast<int>(AlienCellFunctionConstructor::CONSTR::IN_CELL_BRANCH_NO))+"]");
    meta->addSymbolEntry("CONSTR_IN_CELL_FUNCTION","["+QString::number(static_cast<int>(AlienCellFunctionConstructor::CONSTR::IN_CELL_FUNCTION))+"]");
    meta->addSymbolEntry("CONSTR_IN_CELL_FUNCTION::COMPUTER",QString::number(static_cast<int>(AlienCellFunctionConstructor::CONSTR_IN_CELL_FUNCTION::COMPUTER)));
    meta->addSymbolEntry("CONSTR_IN_CELL_FUNCTION::PROP",QString::number(static_cast<int>(AlienCellFunctionConstructor::CONSTR_IN_CELL_FUNCTION::PROP)));
    meta->addSymbolEntry("CONSTR_IN_CELL_FUNCTION::SCANNER",QString::number(static_cast<int>(AlienCellFunctionConstructor::CONSTR_IN_CELL_FUNCTION::SCANNER)));
    meta->addSymbolEntry("CONSTR_IN_CELL_FUNCTION::WEAPON",QString::number(static_cast<int>(AlienCellFunctionConstructor::CONSTR_IN_CELL_FUNCTION::WEAPON)));
    meta->addSymbolEntry("CONSTR_IN_CELL_FUNCTION::CONSTR",QString::number(static_cast<int>(AlienCellFunctionConstructor::CONSTR_IN_CELL_FUNCTION::CONSTR)));
    meta->addSymbolEntry("CONSTR_IN_CELL_FUNCTION::SENSOR",QString::number(static_cast<int>(AlienCellFunctionConstructor::CONSTR_IN_CELL_FUNCTION::SENSOR)));
    meta->addSymbolEntry("CONSTR_IN_CELL_FUNCTION::COMMUNICATOR",QString::number(static_cast<int>(AlienCellFunctionConstructor::CONSTR_IN_CELL_FUNCTION::COMMUNICATOR)));
    meta->addSymbolEntry("CONSTR_IN_CELL_FUNCTION_DATA","["+QString::number(static_cast<int>(AlienCellFunctionConstructor::CONSTR::IN_CELL_FUNCTION_DATA))+"]");

    //propulsion
    meta->addSymbolEntry("PROP_OUT","["+QString::number(static_cast<int>(AlienCellFunctionPropulsion::PROP::OUT))+"]");
    meta->addSymbolEntry("PROP_OUT::SUCCESS",QString::number(static_cast<int>(AlienCellFunctionPropulsion::PROP_OUT::SUCCESS)));
    meta->addSymbolEntry("PROP_OUT::SUCCESS_DAMPING_FINISHED",QString::number(static_cast<int>(AlienCellFunctionPropulsion::PROP_OUT::SUCCESS_DAMPING_FINISHED)));
    meta->addSymbolEntry("PROP_OUT::ERROR_NO_ENERGY",QString::number(static_cast<int>(AlienCellFunctionPropulsion::PROP_OUT::ERROR_NO_ENERGY)));
    meta->addSymbolEntry("PROP_IN","["+QString::number(static_cast<int>(AlienCellFunctionPropulsion::PROP::IN))+"]");
    meta->addSymbolEntry("PROP_IN::DO_NOTHING",QString::number(static_cast<int>(AlienCellFunctionPropulsion::PROP_IN::DO_NOTHING)));
    meta->addSymbolEntry("PROP_IN::BY_ANGLE",QString::number(static_cast<int>(AlienCellFunctionPropulsion::PROP_IN::BY_ANGLE)));
    meta->addSymbolEntry("PROP_IN::FROM_CENTER",QString::number(static_cast<int>(AlienCellFunctionPropulsion::PROP_IN::FROM_CENTER)));
    meta->addSymbolEntry("PROP_IN::TOWARD_CENTER",QString::number(static_cast<int>(AlienCellFunctionPropulsion::PROP_IN::TOWARD_CENTER)));
    meta->addSymbolEntry("PROP_IN::ROTATION_CLOCKWISE",QString::number(static_cast<int>(AlienCellFunctionPropulsion::PROP_IN::ROTATION_CLOCKWISE)));
    meta->addSymbolEntry("PROP_IN::ROTATION_COUNTERCLOCKWISE",QString::number(static_cast<int>(AlienCellFunctionPropulsion::PROP_IN::ROTATION_COUNTERCLOCKWISE)));
    meta->addSymbolEntry("PROP_IN::DAMP_ROTATION",QString::number(static_cast<int>(AlienCellFunctionPropulsion::PROP_IN::DAMP_ROTATION)));
    meta->addSymbolEntry("PROP_IN_ANGLE","["+QString::number(static_cast<int>(AlienCellFunctionPropulsion::PROP::IN_ANGLE))+"]");
    meta->addSymbolEntry("PROP_IN_POWER","["+QString::number(static_cast<int>(AlienCellFunctionPropulsion::PROP::IN_POWER))+"]");

    //scanner
    meta->addSymbolEntry("SCANNER_OUT","["+QString::number(static_cast<int>(AlienCellFunctionScanner::SCANNER::OUT))+"]");
    meta->addSymbolEntry("SCANNER_OUT::SUCCESS",QString::number(static_cast<int>(AlienCellFunctionScanner::SCANNER_OUT::SUCCESS)));
    meta->addSymbolEntry("SCANNER_OUT::FINISHED",QString::number(static_cast<int>(AlienCellFunctionScanner::SCANNER_OUT::FINISHED)));
    meta->addSymbolEntry("SCANNER_OUT::RESTART",QString::number(static_cast<int>(AlienCellFunctionScanner::SCANNER_OUT::RESTART)));
//    meta->addSymbolEntry("SCANNER_IN","[11]");
    meta->addSymbolEntry("SCANNER_INOUT_CELL_NUMBER","["+QString::number(static_cast<int>(AlienCellFunctionScanner::SCANNER::INOUT_CELL_NUMBER))+"]");
    meta->addSymbolEntry("SCANNER_OUT_MASS","["+QString::number(static_cast<int>(AlienCellFunctionScanner::SCANNER::OUT_MASS))+"]");
    meta->addSymbolEntry("SCANNER_OUT_ENERGY","["+QString::number(static_cast<int>(AlienCellFunctionScanner::SCANNER::OUT_ENERGY))+"]");
    meta->addSymbolEntry("SCANNER_OUT_ANGLE","["+QString::number(static_cast<int>(AlienCellFunctionScanner::SCANNER::OUT_ANGLE))+"]");
    meta->addSymbolEntry("SCANNER_OUT_DIST","["+QString::number(static_cast<int>(AlienCellFunctionScanner::SCANNER::OUT_DIST))+"]");
    meta->addSymbolEntry("SCANNER_OUT_CELL_MAX_CONNECTIONS","["+QString::number(static_cast<int>(AlienCellFunctionScanner::SCANNER::OUT_CELL_MAX_CONNECTIONS))+"]");
    meta->addSymbolEntry("SCANNER_OUT_CELL_BRANCH_NO","["+QString::number(static_cast<int>(AlienCellFunctionScanner::SCANNER::OUT_CELL_BRANCH_NO))+"]");
    meta->addSymbolEntry("SCANNER_OUT_CELL_FUNCTION","["+QString::number(static_cast<int>(AlienCellFunctionScanner::SCANNER::OUT_CELL_FUNCTION))+"]");
    meta->addSymbolEntry("SCANNER_OUT_CELL_FUNCTION::COMPUTER",QString::number(static_cast<int>(AlienCellFunctionScanner::SCANNER_OUT_CELL_FUNCTION::COMPUTER)));
    meta->addSymbolEntry("SCANNER_OUT_CELL_FUNCTION::PROP",QString::number(static_cast<int>(AlienCellFunctionScanner::SCANNER_OUT_CELL_FUNCTION::PROP)));
    meta->addSymbolEntry("SCANNER_OUT_CELL_FUNCTION::SCANNER",QString::number(static_cast<int>(AlienCellFunctionScanner::SCANNER_OUT_CELL_FUNCTION::SCANNER)));
    meta->addSymbolEntry("SCANNER_OUT_CELL_FUNCTION::WEAPON",QString::number(static_cast<int>(AlienCellFunctionScanner::SCANNER_OUT_CELL_FUNCTION::WEAPON)));
    meta->addSymbolEntry("SCANNER_OUT_CELL_FUNCTION::CONSTR",QString::number(static_cast<int>(AlienCellFunctionScanner::SCANNER_OUT_CELL_FUNCTION::CONSTR)));
    meta->addSymbolEntry("SCANNER_OUT_CELL_FUNCTION::SENSOR",QString::number(static_cast<int>(AlienCellFunctionScanner::SCANNER_OUT_CELL_FUNCTION::SENSOR)));
    meta->addSymbolEntry("SCANNER_OUT_CELL_FUNCTION::COMMUNICATOR",QString::number(static_cast<int>(AlienCellFunctionScanner::SCANNER_OUT_CELL_FUNCTION::COMMUNICATOR)));
    meta->addSymbolEntry("SCANNER_OUT_CELL_FUNCTION_DATA","["+QString::number(static_cast<int>(AlienCellFunctionScanner::SCANNER::OUT_CELL_FUNCTION_DATA))+"]");

    //weapon
    meta->addSymbolEntry("WEAPON_OUT","["+QString::number(static_cast<int>(AlienCellFunctionWeapon::WEAPON::OUT))+"]");
    meta->addSymbolEntry("WEAPON_OUT::NO_TARGET",QString::number(static_cast<int>(AlienCellFunctionWeapon::WEAPON_OUT::NO_TARGET)));
    meta->addSymbolEntry("WEAPON_OUT::STRIKE_SUCCESSFUL",QString::number(static_cast<int>(AlienCellFunctionWeapon::WEAPON_OUT::STRIKE_SUCCESSFUL)));

    //sensor
    meta->addSymbolEntry("SENSOR_OUT", "["+QString::number(static_cast<int>(AlienCellFunctionSensor::SENSOR::OUT))+"]");
    meta->addSymbolEntry("SENSOR_OUT::NOTHING_FOUND", QString::number(static_cast<int>(AlienCellFunctionSensor::SENSOR_OUT::NOTHING_FOUND)));
    meta->addSymbolEntry("SENSOR_OUT::CLUSTER_FOUND", QString::number(static_cast<int>(AlienCellFunctionSensor::SENSOR_OUT::CLUSTER_FOUND)));
    meta->addSymbolEntry("SENSOR_IN", "["+QString::number(static_cast<int>(AlienCellFunctionSensor::SENSOR::IN))+"]");
    meta->addSymbolEntry("SENSOR_IN::DO_NOTHING", QString::number(static_cast<int>(AlienCellFunctionSensor::SENSOR_IN::DO_NOTHING)));
    meta->addSymbolEntry("SENSOR_IN::SEARCH_VICINITY", QString::number(static_cast<int>(AlienCellFunctionSensor::SENSOR_IN::SEARCH_VICINITY)));
    meta->addSymbolEntry("SENSOR_IN::SEARCH_BY_ANGLE", QString::number(static_cast<int>(AlienCellFunctionSensor::SENSOR_IN::SEARCH_BY_ANGLE)));
    meta->addSymbolEntry("SENSOR_IN::SEARCH_FROM_CENTER", QString::number(static_cast<int>(AlienCellFunctionSensor::SENSOR_IN::SEARCH_FROM_CENTER)));
    meta->addSymbolEntry("SENSOR_IN::SEARCH_TOWARD_CENTER", QString::number(static_cast<int>(AlienCellFunctionSensor::SENSOR_IN::SEARCH_TOWARD_CENTER)));
    meta->addSymbolEntry("SENSOR_INOUT_ANGLE", "["+QString::number(static_cast<int>(AlienCellFunctionSensor::SENSOR::INOUT_ANGLE))+"]");
    meta->addSymbolEntry("SENSOR_IN_MIN_MASS", "["+QString::number(static_cast<int>(AlienCellFunctionSensor::SENSOR::IN_MIN_MASS))+"]");
    meta->addSymbolEntry("SENSOR_IN_MAX_MASS", "["+QString::number(static_cast<int>(AlienCellFunctionSensor::SENSOR::IN_MAX_MASS))+"]");
    meta->addSymbolEntry("SENSOR_OUT_MASS", "["+QString::number(static_cast<int>(AlienCellFunctionSensor::SENSOR::OUT_MASS))+"]");
    meta->addSymbolEntry("SENSOR_OUT_DIST", "["+QString::number(static_cast<int>(AlienCellFunctionSensor::SENSOR::OUT_DIST))+"]");

    //communicator
    meta->addSymbolEntry("COMMUNICATOR_IN", "["+QString::number(static_cast<int>(AlienCellFunctionCommunicator::COMMUNICATOR::IN))+"]");
    meta->addSymbolEntry("COMMUNICATOR_IN::DO_NOTHING", QString::number(static_cast<int>(AlienCellFunctionCommunicator::COMMUNICATOR_IN::DO_NOTHING)));
    meta->addSymbolEntry("COMMUNICATOR_IN::SET_LISTENING_CHANNEL", QString::number(static_cast<int>(AlienCellFunctionCommunicator::COMMUNICATOR_IN::SET_LISTENING_CHANNEL)));
    meta->addSymbolEntry("COMMUNICATOR_IN::SEND_MESSAGE", QString::number(static_cast<int>(AlienCellFunctionCommunicator::COMMUNICATOR_IN::SEND_MESSAGE)));
    meta->addSymbolEntry("COMMUNICATOR_IN::RECEIVE_MESSAGE", QString::number(static_cast<int>(AlienCellFunctionCommunicator::COMMUNICATOR_IN::RECEIVE_MESSAGE)));
    meta->addSymbolEntry("COMMUNICATOR_IN_CHANNEL", "["+QString::number(static_cast<int>(AlienCellFunctionCommunicator::COMMUNICATOR::IN_CHANNEL))+"]");
    meta->addSymbolEntry("COMMUNICATOR_IN_MESSAGE", "["+QString::number(static_cast<int>(AlienCellFunctionCommunicator::COMMUNICATOR::IN_MESSAGE))+"]");
    meta->addSymbolEntry("COMMUNICATOR_IN_ANGLE", "["+QString::number(static_cast<int>(AlienCellFunctionCommunicator::COMMUNICATOR::IN_ANGLE))+"]");
    meta->addSymbolEntry("COMMUNICATOR_IN_DISTANCE", "["+QString::number(static_cast<int>(AlienCellFunctionCommunicator::COMMUNICATOR::IN_DISTANCE))+"]");
    meta->addSymbolEntry("COMMUNICATOR_OUT_SENT_NUM_MESSAGE", "["+QString::number(static_cast<int>(AlienCellFunctionCommunicator::COMMUNICATOR::OUT_SENT_NUM_MESSAGE))+"]");
    meta->addSymbolEntry("COMMUNICATOR_OUT_RECEIVED_NEW_MESSAGE", "["+QString::number(static_cast<int>(AlienCellFunctionCommunicator::COMMUNICATOR::OUT_RECEIVED_NEW_MESSAGE))+"]");
    meta->addSymbolEntry("COMMUNICATOR_OUT_RECEIVED_MESSAGE", "["+QString::number(static_cast<int>(AlienCellFunctionCommunicator::COMMUNICATOR::OUT_RECEIVED_MESSAGE))+"]");
    meta->addSymbolEntry("COMMUNICATOR_OUT_RECEIVED_ANGLE", "["+QString::number(static_cast<int>(AlienCellFunctionCommunicator::COMMUNICATOR::OUT_RECEIVED_ANGLE))+"]");
    meta->addSymbolEntry("COMMUNICATOR_OUT_RECEIVED_DISTANCE", "["+QString::number(static_cast<int>(AlienCellFunctionCommunicator::COMMUNICATOR::OUT_RECEIVED_DISTANCE))+"]");
    meta->addSymbolEntry("COMMUNICATOR_OUT_RECEIVED_NEW_MESSAGE::NO_NEW_MESSAGE", QString::number(static_cast<int>(AlienCellFunctionCommunicator::COMMUNICATOR_OUT_RECEIVED_NEW_MESSAGE::NO_NEW_MESSAGE)));
    meta->addSymbolEntry("COMMUNICATOR_OUT_RECEIVED_NEW_MESSAGE::NEW_MESSAGE", QString::number(static_cast<int>(AlienCellFunctionCommunicator::COMMUNICATOR_OUT_RECEIVED_NEW_MESSAGE::NEW_MESSAGE)));
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
      CELL_CODESIZE(15),
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
    stream << CELL_CODESIZE;
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
    stream >> CELL_FUNCTION_COMMUNICATOR_RANGE;
    stream >> NEW_TOKEN_ENERGY;
    stream >> MIN_TOKEN_ENERGY;
    stream >> RAD_EXPONENT;
    stream >> RAD_FACTOR;
    stream >> RAD_PROBABILITY;
    stream >> CELL_RAD_ENERGY_VEL_MULT;
    stream >> CELL_RAD_ENERGY_VEL_PERTURB;
}
