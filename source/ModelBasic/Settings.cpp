#include <QtGlobal>

#include "SymbolTable.h"
#include "SimulationParameters.h"
#include "Settings.h"

SymbolTable* ModelSettings::getDefaultSymbolTable()
{
	SymbolTable* symbolTable = new SymbolTable();
    symbolTable->clear();

    //general variables
    symbolTable->addEntry("i","[255]");
    symbolTable->addEntry("j","[254]");
    symbolTable->addEntry("k","[253]");
    symbolTable->addEntry("l","[252]");

    //token branch number
    symbolTable->addEntry("BRANCH_NUMBER","[0]");

    //energy guidance system
    symbolTable->addEntry("ENERGY_GUIDANCE_IN","["+std::to_string(Enums::EnergyGuidance::INPUT)+"]");
    symbolTable->addEntry("ENERGY_GUIDANCE_IN::DEACTIVATED",std::to_string(Enums::EnergyGuidanceIn::DEACTIVATED));
    symbolTable->addEntry("ENERGY_GUIDANCE_IN::BALANCE_CELL",std::to_string(Enums::EnergyGuidanceIn::BALANCE_CELL));
    symbolTable->addEntry("ENERGY_GUIDANCE_IN::BALANCE_TOKEN",std::to_string(Enums::EnergyGuidanceIn::BALANCE_TOKEN));
    symbolTable->addEntry("ENERGY_GUIDANCE_IN::BALANCE_BOTH",std::to_string(Enums::EnergyGuidanceIn::BALANCE_BOTH));
    symbolTable->addEntry("ENERGY_GUIDANCE_IN::HARVEST_CELL",std::to_string(Enums::EnergyGuidanceIn::HARVEST_CELL));
    symbolTable->addEntry("ENERGY_GUIDANCE_IN::HARVEST_TOKEN",std::to_string(Enums::EnergyGuidanceIn::HARVEST_TOKEN));
    symbolTable->addEntry("ENERGY_GUIDANCE_IN_VALUE_CELL","["+std::to_string(Enums::EnergyGuidance::IN_VALUE_CELL)+"]");
    symbolTable->addEntry("ENERGY_GUIDANCE_IN_VALUE_TOKEN","["+std::to_string(Enums::EnergyGuidance::IN_VALUE_TOKEN)+"]");

    //constructor
    symbolTable->addEntry("CONSTR_OUT","["+std::to_string(Enums::Constr::OUTPUT)+"]");
    symbolTable->addEntry("CONSTR_OUT::SUCCESS",std::to_string(Enums::ConstrOut::SUCCESS));
    symbolTable->addEntry("CONSTR_OUT::SUCCESS_ROT",std::to_string(Enums::ConstrOut::SUCCESS_ROT));
    symbolTable->addEntry("CONSTR_OUT::ERROR_NO_ENERGY",std::to_string(Enums::ConstrOut::ERROR_NO_ENERGY));
    symbolTable->addEntry("CONSTR_OUT::ERROR_OBSTACLE",std::to_string(Enums::ConstrOut::ERROR_OBSTACLE));
    symbolTable->addEntry("CONSTR_OUT::ERROR_CONNECTION",std::to_string(Enums::ConstrOut::ERROR_CONNECTION));
    symbolTable->addEntry("CONSTR_OUT::ERROR_DIST",std::to_string(Enums::ConstrOut::ERROR_DIST));
    symbolTable->addEntry("CONSTR_IN","["+std::to_string(Enums::Constr::INPUT)+"]");
    symbolTable->addEntry("CONSTR_IN::DO_NOTHING",std::to_string(Enums::ConstrIn::DO_NOTHING));
    symbolTable->addEntry("CONSTR_IN::SAFE",std::to_string(Enums::ConstrIn::SAFE));
    symbolTable->addEntry("CONSTR_IN::UNSAFE",std::to_string(Enums::ConstrIn::UNSAFE));
    symbolTable->addEntry("CONSTR_IN::BRUTEFORCE",std::to_string(Enums::ConstrIn::BRUTEFORCE));
    symbolTable->addEntry("CONSTR_IN_OPTION","["+std::to_string(Enums::Constr::IN_OPTION)+"]");
    symbolTable->addEntry("CONSTR_IN_OPTION::STANDARD",std::to_string(Enums::ConstrInOption::STANDARD));
    symbolTable->addEntry("CONSTR_IN_OPTION::CREATE_EMPTY_TOKEN",std::to_string(Enums::ConstrInOption::CREATE_EMPTY_TOKEN));
    symbolTable->addEntry("CONSTR_IN_OPTION::CREATE_DUP_TOKEN",std::to_string(Enums::ConstrInOption::CREATE_DUP_TOKEN));
    symbolTable->addEntry("CONSTR_IN_OPTION::FINISH_NO_SEP",std::to_string(Enums::ConstrInOption::FINISH_NO_SEP));
    symbolTable->addEntry("CONSTR_IN_OPTION::FINISH_WITH_SEP",std::to_string(Enums::ConstrInOption::FINISH_WITH_SEP));
    symbolTable->addEntry("CONSTR_IN_OPTION::FINISH_WITH_SEP_RED",std::to_string(Enums::ConstrInOption::FINISH_WITH_SEP_RED));
    symbolTable->addEntry("CONSTR_IN_OPTION::FINISH_WITH_TOKEN_SEP_RED",std::to_string(Enums::ConstrInOption::FINISH_WITH_TOKEN_SEP_RED));
    symbolTable->addEntry("CONSTR_INOUT_ANGLE","["+std::to_string(Enums::Constr::INOUT_ANGLE)+"]");
    symbolTable->addEntry("CONSTR_IN_DIST","["+std::to_string(Enums::Constr::IN_DIST)+"]");
    symbolTable->addEntry("CONSTR_IN_CELL_MAX_CONNECTIONS","["+std::to_string(Enums::Constr::IN_CELL_MAX_CONNECTIONS)+"]");
    symbolTable->addEntry("CONSTR_IN_CELL_MAX_CONNECTIONS::AUTO","0");       //artificial entry (has no symbol in enum class)
    symbolTable->addEntry("CONSTR_IN_CELL_BRANCH_NO","["+std::to_string(Enums::Constr::IN_CELL_BRANCH_NO)+"]");
    symbolTable->addEntry("CONSTR_IN_CELL_FUNCTION","["+std::to_string(Enums::Constr::IN_CELL_FUNCTION)+"]");
    symbolTable->addEntry("CONSTR_IN_CELL_FUNCTION::COMPUTER",std::to_string(Enums::CellFunction::COMPUTER));
    symbolTable->addEntry("CONSTR_IN_CELL_FUNCTION::PROP",std::to_string(Enums::CellFunction::PROPULSION));
    symbolTable->addEntry("CONSTR_IN_CELL_FUNCTION::SCANNER",std::to_string(Enums::CellFunction::SCANNER));
    symbolTable->addEntry("CONSTR_IN_CELL_FUNCTION::WEAPON",std::to_string(Enums::CellFunction::WEAPON));
    symbolTable->addEntry("CONSTR_IN_CELL_FUNCTION::CONSTR",std::to_string(Enums::CellFunction::CONSTRUCTOR));
    symbolTable->addEntry("CONSTR_IN_CELL_FUNCTION::SENSOR",std::to_string(Enums::CellFunction::SENSOR));
    symbolTable->addEntry("CONSTR_IN_CELL_FUNCTION::COMMUNICATOR",std::to_string(Enums::CellFunction::COMMUNICATOR));
    symbolTable->addEntry("CONSTR_IN_CELL_FUNCTION_DATA","["+std::to_string(Enums::Constr::IN_CELL_FUNCTION_DATA)+"]");

    //propulsion
    symbolTable->addEntry("PROP_OUT","["+std::to_string(Enums::Prop::OUTPUT)+"]");
    symbolTable->addEntry("PROP_OUT::SUCCESS",std::to_string(Enums::PropOut::SUCCESS));
    symbolTable->addEntry("PROP_OUT::SUCCESS_DAMPING_FINISHED",std::to_string(Enums::PropOut::SUCCESS_DAMPING_FINISHED));
    symbolTable->addEntry("PROP_OUT::ERROR_NO_ENERGY",std::to_string(Enums::PropOut::ERROR_NO_ENERGY));
    symbolTable->addEntry("PROP_IN","["+std::to_string(Enums::Prop::INPUT)+"]");
    symbolTable->addEntry("PROP_IN::DO_NOTHING",std::to_string(Enums::PropIn::DO_NOTHING));
    symbolTable->addEntry("PROP_IN::BY_ANGLE",std::to_string(Enums::PropIn::BY_ANGLE));
    symbolTable->addEntry("PROP_IN::FROM_CENTER",std::to_string(Enums::PropIn::FROM_CENTER));
    symbolTable->addEntry("PROP_IN::TOWARD_CENTER",std::to_string(Enums::PropIn::TOWARD_CENTER));
    symbolTable->addEntry("PROP_IN::ROTATION_CLOCKWISE",std::to_string(Enums::PropIn::ROTATION_CLOCKWISE));
    symbolTable->addEntry("PROP_IN::ROTATION_COUNTERCLOCKWISE",std::to_string(Enums::PropIn::ROTATION_COUNTERCLOCKWISE));
    symbolTable->addEntry("PROP_IN::DAMP_ROTATION",std::to_string(Enums::PropIn::DAMP_ROTATION));
    symbolTable->addEntry("PROP_IN_ANGLE","["+std::to_string(Enums::Prop::IN_ANGLE)+"]");
    symbolTable->addEntry("PROP_IN_POWER","["+std::to_string(Enums::Prop::IN_POWER)+"]");

    //scanner
    symbolTable->addEntry("SCANNER_OUT","["+std::to_string(Enums::Scanner::OUTPUT)+"]");
    symbolTable->addEntry("SCANNER_OUT::SUCCESS",std::to_string(Enums::ScannerOut::SUCCESS));
    symbolTable->addEntry("SCANNER_OUT::FINISHED",std::to_string(Enums::ScannerOut::FINISHED));
    symbolTable->addEntry("SCANNER_OUT::RESTART",std::to_string(Enums::ScannerOut::RESTART));
//    meta->addEntry("SCANNER_IN","[11]");
    symbolTable->addEntry("SCANNER_INOUT_CELL_NUMBER","["+std::to_string(Enums::Scanner::INOUT_CELL_NUMBER)+"]");
    symbolTable->addEntry("SCANNER_OUT_MASS","["+std::to_string(Enums::Scanner::OUT_MASS)+"]");
    symbolTable->addEntry("SCANNER_OUT_ENERGY","["+std::to_string(Enums::Scanner::OUT_ENERGY)+"]");
    symbolTable->addEntry("SCANNER_OUT_ANGLE","["+std::to_string(Enums::Scanner::OUT_ANGLE)+"]");
    symbolTable->addEntry("SCANNER_OUT_DISTANCE","["+std::to_string(Enums::Scanner::OUT_DISTANCE)+"]");
    symbolTable->addEntry("SCANNER_OUT_CELL_MAX_CONNECTIONS","["+std::to_string(Enums::Scanner::OUT_CELL_MAX_CONNECTIONS)+"]");
    symbolTable->addEntry("SCANNER_OUT_CELL_BRANCH_NO","["+std::to_string(Enums::Scanner::OUT_CELL_BRANCH_NO)+"]");
    symbolTable->addEntry("SCANNER_OUT_CELL_FUNCTION","["+std::to_string(Enums::Scanner::OUT_CELL_FUNCTION)+"]");
    symbolTable->addEntry("SCANNER_OUT_CELL_FUNCTION::COMPUTER",std::to_string(Enums::CellFunction::COMPUTER));
    symbolTable->addEntry("SCANNER_OUT_CELL_FUNCTION::PROP",std::to_string(Enums::CellFunction::PROPULSION));
    symbolTable->addEntry("SCANNER_OUT_CELL_FUNCTION::SCANNER",std::to_string(Enums::CellFunction::SCANNER));
    symbolTable->addEntry("SCANNER_OUT_CELL_FUNCTION::WEAPON",std::to_string(Enums::CellFunction::WEAPON));
    symbolTable->addEntry("SCANNER_OUT_CELL_FUNCTION::CONSTR",std::to_string(Enums::CellFunction::CONSTRUCTOR));
    symbolTable->addEntry("SCANNER_OUT_CELL_FUNCTION::SENSOR",std::to_string(Enums::CellFunction::SENSOR));
    symbolTable->addEntry("SCANNER_OUT_CELL_FUNCTION::COMMUNICATOR",std::to_string(Enums::CellFunction::COMMUNICATOR));
    symbolTable->addEntry("SCANNER_OUT_CELL_FUNCTION_DATA","["+std::to_string(Enums::Scanner::OUT_CELL_FUNCTION_DATA)+"]");

    //weapon
    symbolTable->addEntry("WEAPON_OUT","["+std::to_string(Enums::Weapon::OUTPUT)+"]");
    symbolTable->addEntry("WEAPON_OUT::NO_TARGET",std::to_string(Enums::WeaponOut::NO_TARGET));
    symbolTable->addEntry("WEAPON_OUT::STRIKE_SUCCESSFUL",std::to_string(Enums::WeaponOut::STRIKE_SUCCESSFUL));

    //sensor
    symbolTable->addEntry("SENSOR_OUT", "["+std::to_string(Enums::Sensor::OUTPUT)+"]");
    symbolTable->addEntry("SENSOR_OUT::NOTHING_FOUND", std::to_string(Enums::SensorOut::NOTHING_FOUND));
    symbolTable->addEntry("SENSOR_OUT::CLUSTER_FOUND", std::to_string(Enums::SensorOut::CLUSTER_FOUND));
    symbolTable->addEntry("SENSOR_IN", "["+std::to_string(Enums::Sensor::INPUT)+"]");
    symbolTable->addEntry("SENSOR_IN::DO_NOTHING", std::to_string(Enums::SensorIn::DO_NOTHING));
    symbolTable->addEntry("SENSOR_IN::SEARCH_VICINITY", std::to_string(Enums::SensorIn::SEARCH_VICINITY));
    symbolTable->addEntry("SENSOR_IN::SEARCH_BY_ANGLE", std::to_string(Enums::SensorIn::SEARCH_BY_ANGLE));
    symbolTable->addEntry("SENSOR_IN::SEARCH_FROM_CENTER", std::to_string(Enums::SensorIn::SEARCH_FROM_CENTER));
    symbolTable->addEntry("SENSOR_IN::SEARCH_TOWARD_CENTER", std::to_string(Enums::SensorIn::SEARCH_TOWARD_CENTER));
    symbolTable->addEntry("SENSOR_INOUT_ANGLE", "["+std::to_string(Enums::Sensor::INOUT_ANGLE)+"]");
    symbolTable->addEntry("SENSOR_IN_MIN_MASS", "["+std::to_string(Enums::Sensor::IN_MIN_MASS)+"]");
    symbolTable->addEntry("SENSOR_IN_MAX_MASS", "["+std::to_string(Enums::Sensor::IN_MAX_MASS)+"]");
    symbolTable->addEntry("SENSOR_OUT_MASS", "["+std::to_string(Enums::Sensor::OUT_MASS)+"]");
    symbolTable->addEntry("SENSOR_OUT_DISTANCE", "["+std::to_string(Enums::Sensor::OUT_DISTANCE)+"]");

    //communicator
    symbolTable->addEntry("COMMUNICATOR_IN", "["+std::to_string(Enums::Communicator::INPUT)+"]");
    symbolTable->addEntry("COMMUNICATOR_IN::DO_NOTHING", std::to_string(Enums::CommunicatorIn::DO_NOTHING));
    symbolTable->addEntry("COMMUNICATOR_IN::SET_LISTENING_CHANNEL", std::to_string(Enums::CommunicatorIn::SET_LISTENING_CHANNEL));
    symbolTable->addEntry("COMMUNICATOR_IN::SEND_MESSAGE", std::to_string(Enums::CommunicatorIn::SEND_MESSAGE));
    symbolTable->addEntry("COMMUNICATOR_IN::RECEIVE_MESSAGE", std::to_string(Enums::CommunicatorIn::RECEIVE_MESSAGE));
    symbolTable->addEntry("COMMUNICATOR_IN_CHANNEL", "["+std::to_string(Enums::Communicator::IN_CHANNEL)+"]");
    symbolTable->addEntry("COMMUNICATOR_IN_MESSAGE", "["+std::to_string(Enums::Communicator::IN_MESSAGE)+"]");
    symbolTable->addEntry("COMMUNICATOR_IN_ANGLE", "["+std::to_string(Enums::Communicator::IN_ANGLE)+"]");
    symbolTable->addEntry("COMMUNICATOR_IN_DISTANCE", "["+std::to_string(Enums::Communicator::IN_DISTANCE)+"]");
    symbolTable->addEntry("COMMUNICATOR_OUT_SENT_NUM_MESSAGE", "["+std::to_string(Enums::Communicator::OUT_SENT_NUM_MESSAGE)+"]");
    symbolTable->addEntry("COMMUNICATOR_OUT_RECEIVED_NEW_MESSAGE", "["+std::to_string(Enums::Communicator::OUT_RECEIVED_NEW_MESSAGE)+"]");
    symbolTable->addEntry("COMMUNICATOR_OUT_RECEIVED_NEW_MESSAGE::NO", std::to_string(Enums::CommunicatorOutReceivedNewMessage::NO));
    symbolTable->addEntry("COMMUNICATOR_OUT_RECEIVED_NEW_MESSAGE::YES", std::to_string(Enums::CommunicatorOutReceivedNewMessage::YES));
    symbolTable->addEntry("COMMUNICATOR_OUT_RECEIVED_MESSAGE", "["+std::to_string(Enums::Communicator::OUT_RECEIVED_MESSAGE)+"]");
    symbolTable->addEntry("COMMUNICATOR_OUT_RECEIVED_ANGLE", "["+std::to_string(Enums::Communicator::OUT_RECEIVED_ANGLE)+"]");
    symbolTable->addEntry("COMMUNICATOR_OUT_RECEIVED_DISTANCE", "["+std::to_string(Enums::Communicator::OUT_RECEIVED_DISTANCE)+"]");

	return symbolTable;
}

SimulationParameters ModelSettings::getDefaultSimulationParameters()
{
	SimulationParameters parameters;
	parameters.clusterMaxRadius = 40.0f;

	parameters.cellMinDistance = 0.3f;
	parameters.cellMaxDistance = 1.3f;
	parameters.cellMass_Reciprocal = 1;
	parameters.cellMaxForce = 0.8f;
	parameters.cellMaxForceDecayProb = 0.2f;
    parameters.cellMinTokenUsages = 40000;
    parameters.cellTokenUsageDecayProb = 0.000001f;
    parameters.cellMaxBonds = 6;
	parameters.cellMaxToken = 3;
	parameters.cellMaxTokenBranchNumber = 6;
	parameters.cellCreationMaxConnection = 4;
	parameters.cellCreationTokenAccessNumber = 0;
	parameters.cellMinEnergy = 50.0f;
	parameters.cellTransformationProb = 0.2f;
	parameters.cellFusionVelocity = 0.4f;

	parameters.cellFunctionWeaponStrength = 0.1f;
    parameters.cellFunctionWeaponEnergyCost = 0.5f;
    parameters.cellFunctionComputerMaxInstructions = 15;
	parameters.cellFunctionComputerCellMemorySize = 8;
    parameters.cellFunctionConstructorOffspringCellEnergy = 100.0f;
	parameters.cellFunctionConstructorOffspringCellDistance = 1.0f;
    parameters.cellFunctionConstructorOffspringTokenEnergy = 60.0f;
    parameters.cellFunctionConstructorTokenDataMutationProb = 0.002f;
    parameters.cellFunctionConstructorCellDataMutationProb = 0.002f;
    parameters.cellFunctionConstructorCellPropertyMutationProb = 0.005f;
    parameters.cellFunctionConstructorCellStructureMutationProb = 0.005f;
    parameters.cellFunctionSensorRange = 50.0f;
	parameters.cellFunctionCommunicatorRange = 50.0f;

	parameters.tokenMemorySize = 256;
	parameters.tokenMinEnergy = 3.0f;

	parameters.radiationExponent = 1;
	parameters.radiationFactor = 0.0002f;
	parameters.radiationProb = 0.03f;
	parameters.radiationVelocityMultiplier = 1.0f;
	parameters.radiationVelocityPerturbation = 0.5f;

	return parameters;
}

ExecutionParameters ModelSettings::getDefaultExecutionParameters()
{
    ExecutionParameters result;
    result.activateFreezing = false;
    result.freezingTimesteps = 5;
    result.imageGlow = true;
    return result;
}
