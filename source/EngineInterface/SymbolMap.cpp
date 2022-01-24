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
    result.emplace("ENERGY_GUIDANCE_IN", "[" + std::to_string(Enums::EnergyGuidance_Input) + "]");
    result.emplace("ENERGY_GUIDANCE_IN::DEACTIVATED", std::to_string(Enums::EnergyGuidanceIn_Deactivated));
    result.emplace("ENERGY_GUIDANCE_IN::BALANCE_CELL", std::to_string(Enums::EnergyGuidanceIn_BalanceCell));
    result.emplace("ENERGY_GUIDANCE_IN::BALANCE_TOKEN", std::to_string(Enums::EnergyGuidanceIn_BalanceToken));
    result.emplace("ENERGY_GUIDANCE_IN::BALANCE_BOTH", std::to_string(Enums::EnergyGuidanceIn_BalanceBoth));
    result.emplace("ENERGY_GUIDANCE_IN::HARVEST_CELL", std::to_string(Enums::EnergyGuidanceIn_HarvestCell));
    result.emplace("ENERGY_GUIDANCE_IN::HARVEST_TOKEN", std::to_string(Enums::EnergyGuidanceIn_HarvestToken));
    result.emplace("ENERGY_GUIDANCE_IN_VALUE_CELL", "[" + std::to_string(Enums::EnergyGuidance_InValueCell) + "]");
    result.emplace("ENERGY_GUIDANCE_IN_VALUE_TOKEN", "[" + std::to_string(Enums::EnergyGuidance_InValueToken) + "]");

    //constructor
    result.emplace("CONSTR_OUT", "[" + std::to_string(Enums::Constr_Output) + "]");
    result.emplace("CONSTR_OUT::SUCCESS", std::to_string(Enums::ConstrOut_Success));
    result.emplace("CONSTR_OUT::ERROR_NO_ENERGY", std::to_string(Enums::ConstrOut_ErrorNoEnergy));
    result.emplace("CONSTR_OUT::ERROR_CONNECTION", std::to_string(Enums::ConstrOut_ErrorConnection));
    result.emplace("CONSTR_OUT::ERROR_DIST", std::to_string(Enums::ConstrOut_ErrorDist));
    result.emplace("CONSTR_IN", "[" + std::to_string(Enums::Constr_Input) + "]");
    result.emplace("CONSTR_IN::DO_NOTHING", std::to_string(Enums::ConstrIn_DoNothing));
    result.emplace("CONSTR_IN_OPTION", "[" + std::to_string(Enums::Constr_InOption) + "]");
    result.emplace("CONSTR_IN_OPTION::STANDARD", std::to_string(Enums::ConstrInOption_Standard));
    result.emplace("CONSTR_IN_OPTION::CREATE_EMPTY_TOKEN", std::to_string(Enums::ConstrInOption_CreateEmptyToken));
    result.emplace("CONSTR_IN_OPTION::CREATE_DUP_TOKEN", std::to_string(Enums::ConstrInOption_CreateDupToken));
    result.emplace("CONSTR_IN_OPTION::FINISH_NO_SEP", std::to_string(Enums::ConstrInOption_FinishNoSep));
    result.emplace("CONSTR_IN_OPTION::FINISH_WITH_SEP", std::to_string(Enums::ConstrInOption_FinishWithSep));
    result.emplace("CONSTR_IN_OPTION::FINISH_WITH_EMPTY_TOKEN_SEP", std::to_string(Enums::ConstrInOption_FinishWithEmptyTokenSep));
    result.emplace("CONSTR_IN_OPTION::FINISH_WITH_DUP_TOKEN_SEP", std::to_string(Enums::ConstrInOption_FinishWithDupTokenSep));
    result.emplace("CONSTR_INOUT_ANGLE", "[" + std::to_string(Enums::Constr_InOutAngle) + "]");
    result.emplace("CONSTR_IN_DIST", "[" + std::to_string(Enums::Constr_InDist) + "]");
    result.emplace("CONSTR_IN_CELL_MAX_CONNECTIONS", "[" + std::to_string(Enums::Constr_InCellMaxConnections) + "]");
    result.emplace("CONSTR_IN_CELL_MAX_CONNECTIONS::AUTO", "0");  //artificial entry (has no symbol in enum)
    result.emplace("CONSTR_IN_CELL_BRANCH_NO", "[" + std::to_string(Enums::Constr_InCellBranchNum) + "]");
    result.emplace("CONSTR_IN_CELL_FUNCTION", "[" + std::to_string(Enums::Constr_InCellFunction) + "]");
    result.emplace("CONSTR_IN_CELL_FUNCTION::COMPUTER", std::to_string(Enums::CellFunction_Computation));
    result.emplace("CONSTR_IN_CELL_FUNCTION::SCANNER", std::to_string(Enums::CellFunction_Scanner));
    result.emplace("CONSTR_IN_CELL_FUNCTION::DIGESTION", std::to_string(Enums::CellFunction_Digestion));
    result.emplace("CONSTR_IN_CELL_FUNCTION::CONSTR", std::to_string(Enums::CellFunction_Constructor));
    result.emplace("CONSTR_IN_CELL_FUNCTION::SENSOR", std::to_string(Enums::CellFunction_Sensor));
    result.emplace("CONSTR_IN_CELL_FUNCTION_DATA", "[" + std::to_string(Enums::Constr_InCellFunctionData) + "]");

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
    result.emplace("SCANNER_OUT_CELL_FUNCTION::COMPUTER", std::to_string(Enums::CellFunction_Computation));
    result.emplace("SCANNER_OUT_CELL_FUNCTION::PROP", std::to_string(Enums::CellFunction_Propulsion));
    result.emplace("SCANNER_OUT_CELL_FUNCTION::SCANNER", std::to_string(Enums::CellFunction_Scanner));
    result.emplace("SCANNER_OUT_CELL_FUNCTION::DIGESTION", std::to_string(Enums::CellFunction_Digestion));
    result.emplace("SCANNER_OUT_CELL_FUNCTION::CONSTR", std::to_string(Enums::CellFunction_Constructor));
    result.emplace("SCANNER_OUT_CELL_FUNCTION::SENSOR", std::to_string(Enums::CellFunction_Sensor));
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
    result.emplace("COMMUNICATOR_IN", "[" + std::to_string(Enums::Communicator_Input) + "]");
    result.emplace("COMMUNICATOR_IN::DO_NOTHING", std::to_string(Enums::CommunicatorIn_DoNothing));
    result.emplace("COMMUNICATOR_IN::SET_LISTENING_CHANNEL", std::to_string(Enums::CommunicatorIn_SetListeningChannel));
    result.emplace("COMMUNICATOR_IN::SEND_MESSAGE", std::to_string(Enums::CommunicatorIn_SendMessage));
    result.emplace("COMMUNICATOR_IN::RECEIVE_MESSAGE", std::to_string(Enums::CommunicatorIn_ReceiveMessage));
    result.emplace("COMMUNICATOR_IN_CHANNEL", "[" + std::to_string(Enums::Communicator_InChannel) + "]");
    result.emplace("COMMUNICATOR_IN_MESSAGE", "[" + std::to_string(Enums::Communicator_InMessage) + "]");
    result.emplace("COMMUNICATOR_IN_ANGLE", "[" + std::to_string(Enums::Communicator_InAngle) + "]");
    result.emplace("COMMUNICATOR_IN_DISTANCE", "[" + std::to_string(Enums::Communicator_InDistance) + "]");
    result.emplace("COMMUNICATOR_OUT_SENT_NUM_MESSAGE", "[" + std::to_string(Enums::Communicator_OutSentNumMessage) + "]");
    result.emplace("COMMUNICATOR_OUT_RECEIVED_NEW_MESSAGE", "[" + std::to_string(Enums::Communicator_OutReceivedNewMessage) + "]");
    result.emplace("COMMUNICATOR_OUT_RECEIVED_NEW_MESSAGE::NO", std::to_string(Enums::CommunicatorOutReceivedNewMessage_No));
    result.emplace("COMMUNICATOR_OUT_RECEIVED_NEW_MESSAGE::YES", std::to_string(Enums::CommunicatorOutReceivedNewMessage_Yes));
    result.emplace("COMMUNICATOR_OUT_RECEIVED_MESSAGE", "[" + std::to_string(Enums::Communicator_OutReceivedMessage) + "]");
    result.emplace("COMMUNICATOR_OUT_RECEIVED_ANGLE", "[" + std::to_string(Enums::Communicator_OutReceivedAngle) + "]");
    result.emplace("COMMUNICATOR_OUT_RECEIVED_DISTANCE", "[" + std::to_string(Enums::Communicator_OutReceivedDistance) + "]");

    return result;
}