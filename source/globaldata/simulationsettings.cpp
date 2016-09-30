#include "simulationsettings.h"
#include "../simulation/metadatamanager.h"

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
    meta->addSymbolEntry("ENERGY_GUIDANCE_IN","[1]");
//    meta->addSymbolEntry("ENERGY_GUIDANCE_IN","["+QString::number(AlienCellFunction::ENERGY_GUIDANCE::IN)+"]");
    meta->addSymbolEntry("ENERGY_GUIDANCE_IN::DEACTIVATED","0");
    meta->addSymbolEntry("ENERGY_GUIDANCE_IN::BALANCE_CELL","1");
    meta->addSymbolEntry("ENERGY_GUIDANCE_IN::BALANCE_TOKEN","2");
    meta->addSymbolEntry("ENERGY_GUIDANCE_IN::BALANCE_BOTH","3");
    meta->addSymbolEntry("ENERGY_GUIDANCE_IN::HARVEST_CELL","4");
    meta->addSymbolEntry("ENERGY_GUIDANCE_IN::HARVEST_TOKEN","5");
    meta->addSymbolEntry("ENERGY_GUIDANCE_IN_VALUE_CELL","[2]");
    meta->addSymbolEntry("ENERGY_GUIDANCE_IN_VALUE_TOKEN","[3]");

    //constructor
    meta->addSymbolEntry("CONSTR_OUT","[5]");
    meta->addSymbolEntry("CONSTR_OUT::SUCCESS","0");
    meta->addSymbolEntry("CONSTR_OUT::SUCCESS_ROT","1");
    meta->addSymbolEntry("CONSTR_OUT::ERROR_NO_ENERGY","2");
    meta->addSymbolEntry("CONSTR_OUT::ERROR_OBSTACLE","3");
    meta->addSymbolEntry("CONSTR_OUT::ERROR_CONNECTION","4");
    meta->addSymbolEntry("CONSTR_OUT::ERROR_DIST","5");
    meta->addSymbolEntry("CONSTR_IN","[6]");
    meta->addSymbolEntry("CONSTR_IN::DO_NOTHING","0");
    meta->addSymbolEntry("CONSTR_IN::SAFE","1");
    meta->addSymbolEntry("CONSTR_IN::UNSAFE","2");
    meta->addSymbolEntry("CONSTR_IN::BRUTEFORCE","3");
    meta->addSymbolEntry("CONSTR_IN_OPTION","[7]");
    meta->addSymbolEntry("CONSTR_IN_OPTION::STANDARD","0");
    meta->addSymbolEntry("CONSTR_IN_OPTION::CREATE_EMPTY_TOKEN","1");
    meta->addSymbolEntry("CONSTR_IN_OPTION::CREATE_DUP_TOKEN","2");
    meta->addSymbolEntry("CONSTR_IN_OPTION::FINISH_NO_SEP","3");
    meta->addSymbolEntry("CONSTR_IN_OPTION::FINISH_WITH_SEP","4");
    meta->addSymbolEntry("CONSTR_IN_OPTION::FINISH_WITH_SEP_RED","5");
    meta->addSymbolEntry("CONSTR_IN_OPTION::FINISH_WITH_TOKEN_SEP_RED","6");
    meta->addSymbolEntry("CONSTR_INOUT_ANGLE","[15]");
    meta->addSymbolEntry("CONSTR_IN_DIST","[16]");
    meta->addSymbolEntry("CONSTR_IN_CELL_MAX_CONNECTIONS","[17]");
    meta->addSymbolEntry("CONSTR_IN_CELL_MAX_CONNECTIONS::AUTO","0");
    meta->addSymbolEntry("CONSTR_IN_CELL_BRANCH_NO","[18]");
    meta->addSymbolEntry("CONSTR_IN_CELL_FUNCTION","[19]");
    meta->addSymbolEntry("CONSTR_IN_CELL_FUNCTION::COMPUTER","0");
    meta->addSymbolEntry("CONSTR_IN_CELL_FUNCTION::PROP","1");
    meta->addSymbolEntry("CONSTR_IN_CELL_FUNCTION::SCANNER","2");
    meta->addSymbolEntry("CONSTR_IN_CELL_FUNCTION::WEAPON","3");
    meta->addSymbolEntry("CONSTR_IN_CELL_FUNCTION::CONSTR","4");
    meta->addSymbolEntry("CONSTR_IN_CELL_FUNCTION::SENSOR","5");
    meta->addSymbolEntry("CONSTR_IN_CELL_FUNCTION::COMMUNICATOR","6");
    meta->addSymbolEntry("CONSTR_IN_CELL_FUNCTION_DATA","[35]");

    //propulsion
    meta->addSymbolEntry("PROP_OUT","[5]");
    meta->addSymbolEntry("PROP_OUT::SUCCESS","0");
    meta->addSymbolEntry("PROP_OUT::SUCCESS_FINISHED","1");
    meta->addSymbolEntry("PROP_OUT::ERROR_NO_ENERGY","2");
    meta->addSymbolEntry("PROP_IN","[8]");
    meta->addSymbolEntry("PROP_IN::DO_NOTHING","0");
    meta->addSymbolEntry("PROP_IN::BY_ANGLE","1");
    meta->addSymbolEntry("PROP_IN::FROM_CENTER","2");
    meta->addSymbolEntry("PROP_IN::TOWARD_CENTER","3");
    meta->addSymbolEntry("PROP_IN::ROTATION_CLOCKWISE","4");
    meta->addSymbolEntry("PROP_IN::ROTATION_COUNTERCLOCKWISE","5");
    meta->addSymbolEntry("PROP_IN::DAMP_ROTATION","6");
    meta->addSymbolEntry("PROP_IN_ANGLE","[9]");
    meta->addSymbolEntry("PROP_IN_POWER","[10]");

    //scanner
    meta->addSymbolEntry("SCANNER_OUT","[5]");
    meta->addSymbolEntry("SCANNER_OUT::SUCCESS","0");
    meta->addSymbolEntry("SCANNER_OUT::FINISHED","1");
    meta->addSymbolEntry("SCANNER_OUT::RESTART","2");
//    meta->addSymbolEntry("SCANNER_IN","[11]");
    meta->addSymbolEntry("SCANNER_INOUT_CELL_NUMBER","[12]");
    meta->addSymbolEntry("SCANNER_OUT_MASS","[13]");
    meta->addSymbolEntry("SCANNER_OUT_ENERGY","[14]");
    meta->addSymbolEntry("SCANNER_OUT_ANGLE","[15]");
    meta->addSymbolEntry("SCANNER_OUT_DIST","[16]");
    meta->addSymbolEntry("SCANNER_OUT_CELL_MAX_CONNECTIONS","[17]");
    meta->addSymbolEntry("SCANNER_OUT_CELL_BRANCH_NO","[18]");
    meta->addSymbolEntry("SCANNER_OUT_CELL_FUNCTION","[19]");
    meta->addSymbolEntry("SCANNER_OUT_CELL_FUNCTION::COMPUTER","0");
    meta->addSymbolEntry("SCANNER_OUT_CELL_FUNCTION::PROP","1");
    meta->addSymbolEntry("SCANNER_OUT_CELL_FUNCTION::SCANNER","2");
    meta->addSymbolEntry("SCANNER_OUT_CELL_FUNCTION::WEAPON","3");
    meta->addSymbolEntry("SCANNER_OUT_CELL_FUNCTION::CONSTR","4");
    meta->addSymbolEntry("SCANNER_OUT_CELL_FUNCTION::SENSOR","5");
    meta->addSymbolEntry("SCANNER_OUT_CELL_FUNCTION::COMMUNICATOR","6");
    meta->addSymbolEntry("SCANNER_OUT_CELL_FUNCTION_DATA","[35]");

    //weapon
    meta->addSymbolEntry("WEAPON_OUT","[5]");
    meta->addSymbolEntry("WEAPON_OUT::NO_TARGET","0");
    meta->addSymbolEntry("WEAPON_OUT::STRIKE_SUCCESSFUL","1");

    //sensor
    meta->addSymbolEntry("SENSOR_OUT", "[5]");
    meta->addSymbolEntry("SENSOR_OUT::NOTHING_FOUND", "0");
    meta->addSymbolEntry("SENSOR_OUT::CLUSTER_FOUND", "1");
    meta->addSymbolEntry("SENSOR_IN", "[20]");
    meta->addSymbolEntry("SENSOR_IN::DO_NOTHING", "0");
    meta->addSymbolEntry("SENSOR_IN::SEARCH_VICINITY", "1");
    meta->addSymbolEntry("SENSOR_IN::SEARCH_BY_ANGLE", "2");
    meta->addSymbolEntry("SENSOR_IN::SEARCH_FROM_CENTER", "3");
    meta->addSymbolEntry("SENSOR_IN::SEARCH_TOWARD_CENTER", "4");
    meta->addSymbolEntry("SENSOR_INOUT_ANGLE", "[21]");
    meta->addSymbolEntry("SENSOR_IN_MIN_MASS", "[22]");
    meta->addSymbolEntry("SENSOR_IN_MAX_MASS", "[23]");
    meta->addSymbolEntry("SENSOR_OUT_MASS", "[24]");
    meta->addSymbolEntry("SENSOR_OUT_DIST", "[25]");
}

AlienParameters simulationParameters;

AlienParameters::AlienParameters ()
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

void AlienParameters::serializeData (QDataStream& stream)
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

void AlienParameters::readData (QDataStream& stream)
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
