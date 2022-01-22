#include "SymbolMap.h"

//#TODO update

SymbolMap SymbolMapHelper::getDefaultSymbolMap()
{
    SymbolMap result;

    //general variables
    result.emplace("i", "[255]");
    result.emplace("j", "[254]");
    result.emplace("k", "[253]");
    result.emplace("l", "[252]");

    //token branch number
    result.emplace("BRANCH_NUMBER", "[0]");

    //energy guidance system
    result.emplace("ENERGY_GUIDANCE_IN", "[" + std::to_string(Enums::EnergyGuidance::INPUT) + "]");
    result.emplace("ENERGY_GUIDANCE_IN::DEACTIVATED", std::to_string(Enums::EnergyGuidanceIn::DEACTIVATED));
    result.emplace("ENERGY_GUIDANCE_IN::BALANCE_CELL", std::to_string(Enums::EnergyGuidanceIn::BALANCE_CELL));
    result.emplace("ENERGY_GUIDANCE_IN::BALANCE_TOKEN", std::to_string(Enums::EnergyGuidanceIn::BALANCE_TOKEN));
    result.emplace("ENERGY_GUIDANCE_IN::BALANCE_BOTH", std::to_string(Enums::EnergyGuidanceIn::BALANCE_BOTH));
    result.emplace("ENERGY_GUIDANCE_IN::HARVEST_CELL", std::to_string(Enums::EnergyGuidanceIn::HARVEST_CELL));
    result.emplace("ENERGY_GUIDANCE_IN::HARVEST_TOKEN", std::to_string(Enums::EnergyGuidanceIn::HARVEST_TOKEN));
    result.emplace("ENERGY_GUIDANCE_IN_VALUE_CELL", "[" + std::to_string(Enums::EnergyGuidance::IN_VALUE_CELL) + "]");
    result.emplace("ENERGY_GUIDANCE_IN_VALUE_TOKEN", "[" + std::to_string(Enums::EnergyGuidance::IN_VALUE_TOKEN) + "]");

    //constructor
    result.emplace("CONSTR_OUT", "[" + std::to_string(Enums::Constr::OUTPUT) + "]");
    result.emplace("CONSTR_OUT::SUCCESS", std::to_string(Enums::ConstrOut::SUCCESS));
    result.emplace("CONSTR_OUT::ERROR_NO_ENERGY", std::to_string(Enums::ConstrOut::ERROR_NO_ENERGY));
    result.emplace("CONSTR_OUT::ERROR_CONNECTION", std::to_string(Enums::ConstrOut::ERROR_CONNECTION));
    result.emplace("CONSTR_OUT::ERROR_DIST", std::to_string(Enums::ConstrOut::ERROR_DIST));
    result.emplace("CONSTR_IN", "[" + std::to_string(Enums::Constr::INPUT) + "]");
    result.emplace("CONSTR_IN::DO_NOTHING", std::to_string(Enums::ConstrIn::DO_NOTHING));
    result.emplace("CONSTR_IN_OPTION", "[" + std::to_string(Enums::Constr::IN_OPTION) + "]");
    result.emplace("CONSTR_IN_OPTION::STANDARD", std::to_string(Enums::ConstrInOption::STANDARD));
    result.emplace("CONSTR_IN_OPTION::CREATE_EMPTY_TOKEN", std::to_string(Enums::ConstrInOption::CREATE_EMPTY_TOKEN));
    result.emplace("CONSTR_IN_OPTION::CREATE_DUP_TOKEN", std::to_string(Enums::ConstrInOption::CREATE_DUP_TOKEN));
    result.emplace("CONSTR_IN_OPTION::FINISH_NO_SEP", std::to_string(Enums::ConstrInOption::FINISH_NO_SEP));
    result.emplace("CONSTR_IN_OPTION::FINISH_WITH_SEP", std::to_string(Enums::ConstrInOption::FINISH_WITH_SEP));
    result.emplace("CONSTR_IN_OPTION::FINISH_WITH_EMPTY_TOKEN_SEP", std::to_string(Enums::ConstrInOption::FINISH_WITH_EMPTY_TOKEN_SEP));
    result.emplace("CONSTR_IN_OPTION::FINISH_WITH_DUP_TOKEN_SEP", std::to_string(Enums::ConstrInOption::FINISH_WITH_DUP_TOKEN_SEP));
    result.emplace("CONSTR_INOUT_ANGLE", "[" + std::to_string(Enums::Constr::INOUT_ANGLE) + "]");
    result.emplace("CONSTR_IN_DIST", "[" + std::to_string(Enums::Constr::IN_DIST) + "]");
    result.emplace("CONSTR_IN_CELL_MAX_CONNECTIONS", "[" + std::to_string(Enums::Constr::IN_CELL_MAX_CONNECTIONS) + "]");
    result.emplace("CONSTR_IN_CELL_MAX_CONNECTIONS::AUTO", "0");  //artificial entry (has no symbol in enum class)
    result.emplace("CONSTR_IN_CELL_BRANCH_NO", "[" + std::to_string(Enums::Constr::IN_CELL_BRANCH_NO) + "]");
    result.emplace("CONSTR_IN_CELL_FUNCTION", "[" + std::to_string(Enums::Constr::IN_CELL_FUNCTION) + "]");
    result.emplace("CONSTR_IN_CELL_FUNCTION::COMPUTER", std::to_string(Enums::CellFunction::COMPUTATION));
    result.emplace("CONSTR_IN_CELL_FUNCTION::SCANNER", std::to_string(Enums::CellFunction::SCANNER));
    result.emplace("CONSTR_IN_CELL_FUNCTION::DIGESTION", std::to_string(Enums::CellFunction::DIGESTION));
    result.emplace("CONSTR_IN_CELL_FUNCTION::CONSTR", std::to_string(Enums::CellFunction::CONSTRUCTOR));
    result.emplace("CONSTR_IN_CELL_FUNCTION::SENSOR", std::to_string(Enums::CellFunction::SENSOR));
    result.emplace("CONSTR_IN_CELL_FUNCTION_DATA", "[" + std::to_string(Enums::Constr::IN_CELL_FUNCTION_DATA) + "]");

    //scanner
    result.emplace("SCANNER_OUT", "[" + std::to_string(Enums::Scanner::OUTPUT) + "]");
    result.emplace("SCANNER_OUT::SUCCESS", std::to_string(Enums::ScannerOut::SUCCESS));
    result.emplace("SCANNER_OUT::FINISHED", std::to_string(Enums::ScannerOut::FINISHED));
    //    meta->addEntry("SCANNER_IN","[11]");
    result.emplace("SCANNER_INOUT_CELL_NUMBER", "[" + std::to_string(Enums::Scanner::INOUT_CELL_NUMBER) + "]");
    result.emplace("SCANNER_OUT_ENERGY", "[" + std::to_string(Enums::Scanner::OUT_ENERGY) + "]");
    result.emplace("SCANNER_OUT_ANGLE", "[" + std::to_string(Enums::Scanner::OUT_ANGLE) + "]");
    result.emplace("SCANNER_OUT_DISTANCE", "[" + std::to_string(Enums::Scanner::OUT_DISTANCE) + "]");
    result.emplace("SCANNER_OUT_CELL_MAX_CONNECTIONS", "[" + std::to_string(Enums::Scanner::OUT_CELL_MAX_CONNECTIONS) + "]");
    result.emplace("SCANNER_OUT_CELL_BRANCH_NO", "[" + std::to_string(Enums::Scanner::OUT_CELL_BRANCH_NO) + "]");
    result.emplace("SCANNER_OUT_CELL_FUNCTION", "[" + std::to_string(Enums::Scanner::OUT_CELL_FUNCTION) + "]");
    result.emplace("SCANNER_OUT_CELL_FUNCTION::COMPUTER", std::to_string(Enums::CellFunction::COMPUTATION));
    result.emplace("SCANNER_OUT_CELL_FUNCTION::PROP", std::to_string(Enums::CellFunction::PROPULSION));
    result.emplace("SCANNER_OUT_CELL_FUNCTION::SCANNER", std::to_string(Enums::CellFunction::SCANNER));
    result.emplace("SCANNER_OUT_CELL_FUNCTION::DIGESTION", std::to_string(Enums::CellFunction::DIGESTION));
    result.emplace("SCANNER_OUT_CELL_FUNCTION::CONSTR", std::to_string(Enums::CellFunction::CONSTRUCTOR));
    result.emplace("SCANNER_OUT_CELL_FUNCTION::SENSOR", std::to_string(Enums::CellFunction::SENSOR));
    result.emplace("SCANNER_OUT_CELL_FUNCTION_DATA", "[" + std::to_string(Enums::Scanner::OUT_CELL_FUNCTION_DATA) + "]");

    //weapon
    result.emplace("WEAPON_OUT", "[" + std::to_string(Enums::Digestion::OUTPUT) + "]");
    result.emplace("WEAPON_OUT::NO_TARGET", std::to_string(Enums::DigestionOut::NO_TARGET));
    result.emplace("WEAPON_OUT::STRIKE_SUCCESSFUL", std::to_string(Enums::DigestionOut::STRIKE_SUCCESSFUL));

    //sensor
    result.emplace("SENSOR_OUT", "[" + std::to_string(Enums::Sensor::OUTPUT) + "]");
    result.emplace("SENSOR_OUT::NOTHING_FOUND", std::to_string(Enums::SensorOut::NOTHING_FOUND));
    result.emplace("SENSOR_OUT::CLUSTER_FOUND", std::to_string(Enums::SensorOut::CLUSTER_FOUND));
    result.emplace("SENSOR_IN", "[" + std::to_string(Enums::Sensor::INPUT) + "]");
    result.emplace("SENSOR_IN::DO_NOTHING", std::to_string(Enums::SensorIn::DO_NOTHING));
    result.emplace("SENSOR_IN::SEARCH_VICINITY", std::to_string(Enums::SensorIn::SEARCH_VICINITY));
    result.emplace("SENSOR_IN::SEARCH_BY_ANGLE", std::to_string(Enums::SensorIn::SEARCH_BY_ANGLE));
    result.emplace("SENSOR_IN::SEARCH_FROM_CENTER", std::to_string(Enums::SensorIn::SEARCH_FROM_CENTER));
    result.emplace("SENSOR_IN::SEARCH_TOWARD_CENTER", std::to_string(Enums::SensorIn::SEARCH_TOWARD_CENTER));
    result.emplace("SENSOR_INOUT_ANGLE", "[" + std::to_string(Enums::Sensor::INOUT_ANGLE) + "]");
    result.emplace("SENSOR_IN_MIN_MASS", "[" + std::to_string(Enums::Sensor::IN_MIN_MASS) + "]");
    result.emplace("SENSOR_IN_MAX_MASS", "[" + std::to_string(Enums::Sensor::IN_MAX_MASS) + "]");
    result.emplace("SENSOR_OUT_MASS", "[" + std::to_string(Enums::Sensor::OUT_MASS) + "]");
    result.emplace("SENSOR_OUT_DISTANCE", "[" + std::to_string(Enums::Sensor::OUT_DISTANCE) + "]");

    //communicator
    result.emplace("COMMUNICATOR_IN", "[" + std::to_string(Enums::Communicator::INPUT) + "]");
    result.emplace("COMMUNICATOR_IN::DO_NOTHING", std::to_string(Enums::CommunicatorIn::DO_NOTHING));
    result.emplace("COMMUNICATOR_IN::SET_LISTENING_CHANNEL", std::to_string(Enums::CommunicatorIn::SET_LISTENING_CHANNEL));
    result.emplace("COMMUNICATOR_IN::SEND_MESSAGE", std::to_string(Enums::CommunicatorIn::SEND_MESSAGE));
    result.emplace("COMMUNICATOR_IN::RECEIVE_MESSAGE", std::to_string(Enums::CommunicatorIn::RECEIVE_MESSAGE));
    result.emplace("COMMUNICATOR_IN_CHANNEL", "[" + std::to_string(Enums::Communicator::IN_CHANNEL) + "]");
    result.emplace("COMMUNICATOR_IN_MESSAGE", "[" + std::to_string(Enums::Communicator::IN_MESSAGE) + "]");
    result.emplace("COMMUNICATOR_IN_ANGLE", "[" + std::to_string(Enums::Communicator::IN_ANGLE) + "]");
    result.emplace("COMMUNICATOR_IN_DISTANCE", "[" + std::to_string(Enums::Communicator::IN_DISTANCE) + "]");
    result.emplace("COMMUNICATOR_OUT_SENT_NUM_MESSAGE", "[" + std::to_string(Enums::Communicator::OUT_SENT_NUM_MESSAGE) + "]");
    result.emplace("COMMUNICATOR_OUT_RECEIVED_NEW_MESSAGE", "[" + std::to_string(Enums::Communicator::OUT_RECEIVED_NEW_MESSAGE) + "]");
    result.emplace("COMMUNICATOR_OUT_RECEIVED_NEW_MESSAGE::NO", std::to_string(Enums::CommunicatorOutReceivedNewMessage::NO));
    result.emplace("COMMUNICATOR_OUT_RECEIVED_NEW_MESSAGE::YES", std::to_string(Enums::CommunicatorOutReceivedNewMessage::YES));
    result.emplace("COMMUNICATOR_OUT_RECEIVED_MESSAGE", "[" + std::to_string(Enums::Communicator::OUT_RECEIVED_MESSAGE) + "]");
    result.emplace("COMMUNICATOR_OUT_RECEIVED_ANGLE", "[" + std::to_string(Enums::Communicator::OUT_RECEIVED_ANGLE) + "]");
    result.emplace("COMMUNICATOR_OUT_RECEIVED_DISTANCE", "[" + std::to_string(Enums::Communicator::OUT_RECEIVED_DISTANCE) + "]");

    return result;
}