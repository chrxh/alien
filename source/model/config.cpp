#include <QtGlobal>

#include "model/features/cellfeatureconstants.h"
#include "model/metadata/symboltable.h"

#include "config.h"
#include "simulationparameters.h"

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
    symbolTable->addEntry("ENERGY_GUIDANCE_IN","["+QString::number(Enums::EnergyGuidance::IN)+"]");
    symbolTable->addEntry("ENERGY_GUIDANCE_IN::DEACTIVATED",QString::number(Enums::EnergyGuidanceIn::DEACTIVATED));
    symbolTable->addEntry("ENERGY_GUIDANCE_IN::BALANCE_CELL",QString::number(Enums::EnergyGuidanceIn::BALANCE_CELL));
    symbolTable->addEntry("ENERGY_GUIDANCE_IN::BALANCE_TOKEN",QString::number(Enums::EnergyGuidanceIn::BALANCE_TOKEN));
    symbolTable->addEntry("ENERGY_GUIDANCE_IN::BALANCE_BOTH",QString::number(Enums::EnergyGuidanceIn::BALANCE_BOTH));
    symbolTable->addEntry("ENERGY_GUIDANCE_IN::HARVEST_CELL",QString::number(Enums::EnergyGuidanceIn::HARVEST_CELL));
    symbolTable->addEntry("ENERGY_GUIDANCE_IN::HARVEST_TOKEN",QString::number(Enums::EnergyGuidanceIn::HARVEST_TOKEN));
    symbolTable->addEntry("ENERGY_GUIDANCE_IN_VALUE_CELL","["+QString::number(Enums::EnergyGuidance::IN_VALUE_CELL)+"]");
    symbolTable->addEntry("ENERGY_GUIDANCE_IN_VALUE_TOKEN","["+QString::number(Enums::EnergyGuidance::IN_VALUE_TOKEN)+"]");

    //constructor
    symbolTable->addEntry("CONSTR_OUT","["+QString::number(Enums::Constr::OUT)+"]");
    symbolTable->addEntry("CONSTR_OUT::SUCCESS",QString::number(Enums::ConstrOut::SUCCESS));
    symbolTable->addEntry("CONSTR_OUT::SUCCESS_ROT",QString::number(Enums::ConstrOut::SUCCESS_ROT));
    symbolTable->addEntry("CONSTR_OUT::ERROR_NO_ENERGY",QString::number(Enums::ConstrOut::ERROR_NO_ENERGY));
    symbolTable->addEntry("CONSTR_OUT::ERROR_OBSTACLE",QString::number(Enums::ConstrOut::ERROR_OBSTACLE));
    symbolTable->addEntry("CONSTR_OUT::ERROR_CONNECTION",QString::number(Enums::ConstrOut::ERROR_CONNECTION));
    symbolTable->addEntry("CONSTR_OUT::ERROR_DIST",QString::number(Enums::ConstrOut::ERROR_DIST));
    symbolTable->addEntry("CONSTR_IN","["+QString::number(Enums::Constr::IN)+"]");
    symbolTable->addEntry("CONSTR_IN::DO_NOTHING",QString::number(Enums::ConstrIn::DO_NOTHING));
    symbolTable->addEntry("CONSTR_IN::SAFE",QString::number(Enums::ConstrIn::SAFE));
    symbolTable->addEntry("CONSTR_IN::UNSAFE",QString::number(Enums::ConstrIn::UNSAFE));
    symbolTable->addEntry("CONSTR_IN::BRUTEFORCE",QString::number(Enums::ConstrIn::BRUTEFORCE));
    symbolTable->addEntry("CONSTR_IN_OPTION","["+QString::number(Enums::Constr::IN_OPTION)+"]");
    symbolTable->addEntry("CONSTR_IN_OPTION::STANDARD",QString::number(Enums::ConstrInOption::STANDARD));
    symbolTable->addEntry("CONSTR_IN_OPTION::CREATE_EMPTY_TOKEN",QString::number(Enums::ConstrInOption::CREATE_EMPTY_TOKEN));
    symbolTable->addEntry("CONSTR_IN_OPTION::CREATE_DUP_TOKEN",QString::number(Enums::ConstrInOption::CREATE_DUP_TOKEN));
    symbolTable->addEntry("CONSTR_IN_OPTION::FINISH_NO_SEP",QString::number(Enums::ConstrInOption::FINISH_NO_SEP));
    symbolTable->addEntry("CONSTR_IN_OPTION::FINISH_WITH_SEP",QString::number(Enums::ConstrInOption::FINISH_WITH_SEP));
    symbolTable->addEntry("CONSTR_IN_OPTION::FINISH_WITH_SEP_RED",QString::number(Enums::ConstrInOption::FINISH_WITH_SEP_RED));
    symbolTable->addEntry("CONSTR_IN_OPTION::FINISH_WITH_TOKEN_SEP_RED",QString::number(Enums::ConstrInOption::FINISH_WITH_TOKEN_SEP_RED));
    symbolTable->addEntry("CONSTR_INOUT_ANGLE","["+QString::number(Enums::Constr::INOUT_ANGLE)+"]");
    symbolTable->addEntry("CONSTR_IN_DIST","["+QString::number(Enums::Constr::IN_DIST)+"]");
    symbolTable->addEntry("CONSTR_IN_CELL_MAX_CONNECTIONS","["+QString::number(Enums::Constr::IN_CELL_MAX_CONNECTIONS)+"]");
    symbolTable->addEntry("CONSTR_IN_CELL_MAX_CONNECTIONS::AUTO","0");       //artificial entry (has no symbol in enum class)
    symbolTable->addEntry("CONSTR_IN_CELL_BRANCH_NO","["+QString::number(Enums::Constr::IN_CELL_BRANCH_NO)+"]");
    symbolTable->addEntry("CONSTR_IN_CELL_FUNCTION","["+QString::number(Enums::Constr::IN_CELL_FUNCTION)+"]");
    symbolTable->addEntry("CONSTR_IN_CELL_FUNCTION::COMPUTER",QString::number(Enums::CellFunction::COMPUTER));
    symbolTable->addEntry("CONSTR_IN_CELL_FUNCTION::PROP",QString::number(Enums::CellFunction::PROPULSION));
    symbolTable->addEntry("CONSTR_IN_CELL_FUNCTION::SCANNER",QString::number(Enums::CellFunction::SCANNER));
    symbolTable->addEntry("CONSTR_IN_CELL_FUNCTION::WEAPON",QString::number(Enums::CellFunction::WEAPON));
    symbolTable->addEntry("CONSTR_IN_CELL_FUNCTION::CONSTR",QString::number(Enums::CellFunction::CONSTRUCTOR));
    symbolTable->addEntry("CONSTR_IN_CELL_FUNCTION::SENSOR",QString::number(Enums::CellFunction::SENSOR));
    symbolTable->addEntry("CONSTR_IN_CELL_FUNCTION::COMMUNICATOR",QString::number(Enums::CellFunction::COMMUNICATOR));
    symbolTable->addEntry("CONSTR_IN_CELL_FUNCTION_DATA","["+QString::number(Enums::Constr::IN_CELL_FUNCTION_DATA)+"]");

    //propulsion
    symbolTable->addEntry("PROP_OUT","["+QString::number(Enums::Prop::OUT)+"]");
    symbolTable->addEntry("PROP_OUT::SUCCESS",QString::number(Enums::PropOut::SUCCESS));
    symbolTable->addEntry("PROP_OUT::SUCCESS_DAMPING_FINISHED",QString::number(Enums::PropOut::SUCCESS_DAMPING_FINISHED));
    symbolTable->addEntry("PROP_OUT::ERROR_NO_ENERGY",QString::number(Enums::PropOut::ERROR_NO_ENERGY));
    symbolTable->addEntry("PROP_IN","["+QString::number(Enums::Prop::IN)+"]");
    symbolTable->addEntry("PROP_IN::DO_NOTHING",QString::number(Enums::PropIn::DO_NOTHING));
    symbolTable->addEntry("PROP_IN::BY_ANGLE",QString::number(Enums::PropIn::BY_ANGLE));
    symbolTable->addEntry("PROP_IN::FROM_CENTER",QString::number(Enums::PropIn::FROM_CENTER));
    symbolTable->addEntry("PROP_IN::TOWARD_CENTER",QString::number(Enums::PropIn::TOWARD_CENTER));
    symbolTable->addEntry("PROP_IN::ROTATION_CLOCKWISE",QString::number(Enums::PropIn::ROTATION_CLOCKWISE));
    symbolTable->addEntry("PROP_IN::ROTATION_COUNTERCLOCKWISE",QString::number(Enums::PropIn::ROTATION_COUNTERCLOCKWISE));
    symbolTable->addEntry("PROP_IN::DAMP_ROTATION",QString::number(Enums::PropIn::DAMP_ROTATION));
    symbolTable->addEntry("PROP_IN_ANGLE","["+QString::number(Enums::Prop::IN_ANGLE)+"]");
    symbolTable->addEntry("PROP_IN_POWER","["+QString::number(Enums::Prop::IN_POWER)+"]");

    //scanner
    symbolTable->addEntry("SCANNER_OUT","["+QString::number(Enums::Scanner::OUT)+"]");
    symbolTable->addEntry("SCANNER_OUT::SUCCESS",QString::number(Enums::ScannerOut::SUCCESS));
    symbolTable->addEntry("SCANNER_OUT::FINISHED",QString::number(Enums::ScannerOut::FINISHED));
    symbolTable->addEntry("SCANNER_OUT::RESTART",QString::number(Enums::ScannerOut::RESTART));
//    meta->addEntry("SCANNER_IN","[11]");
    symbolTable->addEntry("SCANNER_INOUT_CELL_NUMBER","["+QString::number(Enums::Scanner::INOUT_CELL_NUMBER)+"]");
    symbolTable->addEntry("SCANNER_OUT_MASS","["+QString::number(Enums::Scanner::OUT_MASS)+"]");
    symbolTable->addEntry("SCANNER_OUT_ENERGY","["+QString::number(Enums::Scanner::OUT_ENERGY)+"]");
    symbolTable->addEntry("SCANNER_OUT_ANGLE","["+QString::number(Enums::Scanner::OUT_ANGLE)+"]");
    symbolTable->addEntry("SCANNER_OUT_DISTANCE","["+QString::number(Enums::Scanner::OUT_DISTANCE)+"]");
    symbolTable->addEntry("SCANNER_OUT_CELL_MAX_CONNECTIONS","["+QString::number(Enums::Scanner::OUT_CELL_MAX_CONNECTIONS)+"]");
    symbolTable->addEntry("SCANNER_OUT_CELL_BRANCH_NO","["+QString::number(Enums::Scanner::OUT_CELL_BRANCH_NO)+"]");
    symbolTable->addEntry("SCANNER_OUT_CELL_FUNCTION","["+QString::number(Enums::Scanner::OUT_CELL_FUNCTION)+"]");
    symbolTable->addEntry("SCANNER_OUT_CELL_FUNCTION::COMPUTER",QString::number(Enums::CellFunction::COMPUTER));
    symbolTable->addEntry("SCANNER_OUT_CELL_FUNCTION::PROP",QString::number(Enums::CellFunction::PROPULSION));
    symbolTable->addEntry("SCANNER_OUT_CELL_FUNCTION::SCANNER",QString::number(Enums::CellFunction::SCANNER));
    symbolTable->addEntry("SCANNER_OUT_CELL_FUNCTION::WEAPON",QString::number(Enums::CellFunction::WEAPON));
    symbolTable->addEntry("SCANNER_OUT_CELL_FUNCTION::CONSTR",QString::number(Enums::CellFunction::CONSTRUCTOR));
    symbolTable->addEntry("SCANNER_OUT_CELL_FUNCTION::SENSOR",QString::number(Enums::CellFunction::SENSOR));
    symbolTable->addEntry("SCANNER_OUT_CELL_FUNCTION::COMMUNICATOR",QString::number(Enums::CellFunction::COMMUNICATOR));
    symbolTable->addEntry("SCANNER_OUT_CELL_FUNCTION_DATA","["+QString::number(Enums::Scanner::OUT_CELL_FUNCTION_DATA)+"]");

    //weapon
    symbolTable->addEntry("WEAPON_OUT","["+QString::number(Enums::Weapon::OUT)+"]");
    symbolTable->addEntry("WEAPON_OUT::NO_TARGET",QString::number(Enums::WeaponOut::NO_TARGET));
    symbolTable->addEntry("WEAPON_OUT::STRIKE_SUCCESSFUL",QString::number(Enums::WeaponOut::STRIKE_SUCCESSFUL));

    //sensor
    symbolTable->addEntry("SENSOR_OUT", "["+QString::number(Enums::Sensor::OUT)+"]");
    symbolTable->addEntry("SENSOR_OUT::NOTHING_FOUND", QString::number(Enums::SensorOut::NOTHING_FOUND));
    symbolTable->addEntry("SENSOR_OUT::CLUSTER_FOUND", QString::number(Enums::SensorOut::CLUSTER_FOUND));
    symbolTable->addEntry("SENSOR_IN", "["+QString::number(Enums::Sensor::IN)+"]");
    symbolTable->addEntry("SENSOR_IN::DO_NOTHING", QString::number(Enums::SensorIn::DO_NOTHING));
    symbolTable->addEntry("SENSOR_IN::SEARCH_VICINITY", QString::number(Enums::SensorIn::SEARCH_VICINITY));
    symbolTable->addEntry("SENSOR_IN::SEARCH_BY_ANGLE", QString::number(Enums::SensorIn::SEARCH_BY_ANGLE));
    symbolTable->addEntry("SENSOR_IN::SEARCH_FROM_CENTER", QString::number(Enums::SensorIn::SEARCH_FROM_CENTER));
    symbolTable->addEntry("SENSOR_IN::SEARCH_TOWARD_CENTER", QString::number(Enums::SensorIn::SEARCH_TOWARD_CENTER));
    symbolTable->addEntry("SENSOR_INOUT_ANGLE", "["+QString::number(Enums::Sensor::INOUT_ANGLE)+"]");
    symbolTable->addEntry("SENSOR_IN_MIN_MASS", "["+QString::number(Enums::Sensor::IN_MIN_MASS)+"]");
    symbolTable->addEntry("SENSOR_IN_MAX_MASS", "["+QString::number(Enums::Sensor::IN_MAX_MASS)+"]");
    symbolTable->addEntry("SENSOR_OUT_MASS", "["+QString::number(Enums::Sensor::OUT_MASS)+"]");
    symbolTable->addEntry("SENSOR_OUT_DISTANCE", "["+QString::number(Enums::Sensor::OUT_DISTANCE)+"]");

    //communicator
    symbolTable->addEntry("COMMUNICATOR_IN", "["+QString::number(Enums::Communicator::IN)+"]");
    symbolTable->addEntry("COMMUNICATOR_IN::DO_NOTHING", QString::number(Enums::CommunicatorIn::DO_NOTHING));
    symbolTable->addEntry("COMMUNICATOR_IN::SET_LISTENING_CHANNEL", QString::number(Enums::CommunicatorIn::SET_LISTENING_CHANNEL));
    symbolTable->addEntry("COMMUNICATOR_IN::SEND_MESSAGE", QString::number(Enums::CommunicatorIn::SEND_MESSAGE));
    symbolTable->addEntry("COMMUNICATOR_IN::RECEIVE_MESSAGE", QString::number(Enums::CommunicatorIn::RECEIVE_MESSAGE));
    symbolTable->addEntry("COMMUNICATOR_IN_CHANNEL", "["+QString::number(Enums::Communicator::IN_CHANNEL)+"]");
    symbolTable->addEntry("COMMUNICATOR_IN_MESSAGE", "["+QString::number(Enums::Communicator::IN_MESSAGE)+"]");
    symbolTable->addEntry("COMMUNICATOR_IN_ANGLE", "["+QString::number(Enums::Communicator::IN_ANGLE)+"]");
    symbolTable->addEntry("COMMUNICATOR_IN_DISTANCE", "["+QString::number(Enums::Communicator::IN_DISTANCE)+"]");
    symbolTable->addEntry("COMMUNICATOR_OUT_SENT_NUM_MESSAGE", "["+QString::number(Enums::Communicator::OUT_SENT_NUM_MESSAGE)+"]");
    symbolTable->addEntry("COMMUNICATOR_OUT_RECEIVED_NEW_MESSAGE", "["+QString::number(Enums::Communicator::OUT_RECEIVED_NEW_MESSAGE)+"]");
    symbolTable->addEntry("COMMUNICATOR_OUT_RECEIVED_NEW_MESSAGE::NO", QString::number(Enums::CommunicatorOutReceivedNewMessage::NO));
    symbolTable->addEntry("COMMUNICATOR_OUT_RECEIVED_NEW_MESSAGE::YES", QString::number(Enums::CommunicatorOutReceivedNewMessage::YES));
    symbolTable->addEntry("COMMUNICATOR_OUT_RECEIVED_MESSAGE", "["+QString::number(Enums::Communicator::OUT_RECEIVED_MESSAGE)+"]");
    symbolTable->addEntry("COMMUNICATOR_OUT_RECEIVED_ANGLE", "["+QString::number(Enums::Communicator::OUT_RECEIVED_ANGLE)+"]");
    symbolTable->addEntry("COMMUNICATOR_OUT_RECEIVED_DISTANCE", "["+QString::number(Enums::Communicator::OUT_RECEIVED_DISTANCE)+"]");
}

void Metadata::loadDefaultSimulationParameters(SimulationParameters* parameters)
{
	parameters->cellMutationProb = 0.0;
	parameters->cellMinDistance = 0.3;
	parameters->cellMaxDistance = 1.3;
	parameters->cellMass_Reciprocal = 1;
	parameters->callMaxForce = 0.8;
	parameters->cellMaxForceDecayProb = 0.2;
	parameters->cellMaxBonds = 6;
	parameters->cellMaxToken = 9;
	parameters->cellMaxTokenBranchNumber = 6;
	parameters->cellCreationEnergy = 100.0;
	parameters->NEW_CELL_MAX_CONNECTION = 4;
	parameters->NEW_CELL_TOKEN_ACCESS_NUMBER = 0;
	parameters->cellMinEnergy = 50.0;
	parameters->cellTransformationProb = 0.2;
	parameters->cellFusionVelocity = 0.4;

	parameters->cellFunctionWeaponStrength = 0.1;
	parameters->cellFunctionComputerMaxInstructions = 15;
	parameters->cellFunctionComputerCellMemorySize = 8;
	parameters->cellFunctionComputerTokenMemorySize = 256;
	parameters->cellFunctionConstructorOffspringDistance = 1.0;
	parameters->cellFunctionSensorRange = 100.0;
	parameters->cellFunctionCommunicatorRange = 30.0;

	parameters->tokenCreationEnergy = 60.0;
	parameters->tokenMinEnergy = 3.0;

	parameters->radiationExponent = 1.0;
	parameters->radiationFactor = 0.0001;
	parameters->radiationProb = 0.01;
	parameters->radiationVelocityMultiplier = 1.0;
	parameters->radiationVelocityPerturbation = 0.5;
}
