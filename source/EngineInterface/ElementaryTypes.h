#pragma once

#include <cstdint>

namespace Enums
{
    using Branching = int;
    enum Branching_
    {
        Branching_TokenBranchNumber = 0
    };

    using CellFunction = int;
    enum CellFunction_
    {
        CellFunction_Computation,
        CellFunction_Propulsion,
        CellFunction_Scanner,
        CellFunction_Digestion,
        CellFunction_Constructor,
        CellFunction_Sensor,
        CellFunction_Muscle,
        CellFunction_Count
    };

    using EnergyGuidance = int;
    enum EnergyGuidance_
    {
        EnergyGuidance_Input = 1,
        EnergyGuidance_InValueCell = 2,
        EnergyGuidance_InValueToken = 3
    };

    using EnergyGuidanceIn = int;
    enum EnergyGuidanceIn_
    {
        EnergyGuidanceIn_Deactivated,
        EnergyGuidanceIn_BalanceCell,
        EnergyGuidanceIn_BalanceToken,
        EnergyGuidanceIn_BalanceBoth,
        EnergyGuidanceIn_HarvestCell,
        EnergyGuidanceIn_HarvestToken,
        EnergyGuidanceIn_Count
    };

    using Communicator = int;
    enum Communicator_
    {
        Communicator_Input = 26,
        Communicator_InChannel = 27,
        Communicator_InMessage = 28,
        Communicator_InAngle = 29,
        Communicator_InDistance = 30,
        Communicator_OutSentNumMessage = 31,
        Communicator_OutReceivedNewMessage = 32,
        Communicator_OutReceivedMessage = 33,
        Communicator_OutReceivedAngle = 34,
        Communicator_OutReceivedDistance = 35,
    };
    using CommunicatorIn = int;
    enum CommunicatorIn_
    {
        CommunicatorIn_DoNothing,
        CommunicatorIn_SetListeningChannel,
        CommunicatorIn_SendMessage,
        CommunicatorIn_ReceiveMessage,
        CommunicatorIn_Count
    };
    using CommunicatorOutReceivedNewMessage = int;
    enum CommunicatorOutReceivedNewMessage_
    {
        CommunicatorOutReceivedNewMessage_No,
        CommunicatorOutReceivedNewMessage_Yes
    };

    using ComputationOperation = int;
    enum ComputationOperation_
    {
        ComputationOperation_Mov,
        ComputationOperation_Add,
        ComputationOperation_Sub,
        ComputationOperation_Mul,
        ComputationOperation_Div,
        ComputationOperation_Xor,
        ComputationOperation_Or,
        ComputationOperation_And,
        ComputationOperation_Ifg,
        ComputationOperation_Ifge,
        ComputationOperation_Ife,
        ComputationOperation_Ifne,
        ComputationOperation_Ifle,
        ComputationOperation_Ifl,
        ComputationOperation_Else,
        ComputationOperation_Endif
    };
    using ComputationOpType = int;
    enum ComputationOpType_
    {
        ComputationOpType_Mem,
        ComputationOpType_MemMem,
        ComputationOpType_Cmem,
        ComputationOpType_Constant
    };

    using Constr = int;
    enum Constr_ {
        Constr_Output = 5,
        Constr_Input = 6,
        Constr_InOption = 7,
        Constr_InAngleAlignment = 38,  //0: no alignment, 2: alignment to 180 deg, 3: alignment to 120 deg, ... up to 6
        Constr_InUniformDist = 13,
        Constr_InOutAngle = 15,
        Constr_InDist = 16,
        Constr_InCellMaxConnections = 17,  //0: automatically; >0: max connections (not greater than MAX_CELL_CONNECTIONS)
        Constr_InCellBranchNum = 18,
        Constr_InCellMetadata = 19,
        Constr_InCellFunction = 39,
        Constr_InCellFunctionData = 40
    };
    using ConstrOut = int;
    enum ConstrOut_
    {
        ConstrOut_Success,
        ConstrOut_ErrorNoEnergy,
        ConstrOut_ErrorConnection,
        ConstrOut_ErrorLock,
        ConstrOut_ErrorDist
    };
    using ConstrIn = int;
    enum ConstrIn_
    {
        ConstrIn_DoNothing,
        ConstrIn_Construct,
        ConstrIn_Count
    };
    using ConstrInOption = int;
    enum ConstrInOption_
    {
        ConstrInOption_Standard,
        ConstrInOption_CreateEmptyToken,
        ConstrInOption_CreateDupToken,
        ConstrInOption_FinishNoSep,
        ConstrInOption_FinishWithSep,
        ConstrInOption_FinishWithEmptyTokenSep,
        ConstrInOption_FinishWithDupTokenSep,
        ConstrInOption_Count
    };

    using ConstrInUniformDist = int;
    enum ConstrInUniformDist_
    {
        ConstrInUniformDist_No,
        ConstrInUniformDist_Yes,
        ConstrInUniformDist_Count
    };

    struct Prop {
        enum Type {
            OUTPUT = 5,
            INPUT = 8,
            IN_ANGLE = 9,
            IN_POWER = 10
        };
    };
    struct PropOut {
        enum Type {
            SUCCESS,
            ERROR_NO_ENERGY
        };
    };
    struct PropIn {
        enum Type {
            DO_NOTHING,
            BY_ANGLE,
            DAMP_ROTATION,
            _COUNTER
        };
    };

    struct Scanner {
        enum Type {
            OUTPUT = 5,
            INOUT_CELL_NUMBER = 12,
            OUT_ENERGY = 14,
            OUT_ANGLE = 15,
            OUT_DISTANCE = 16,
            OUT_CELL_MAX_CONNECTIONS = 17,
            OUT_CELL_BRANCH_NO = 18,
            OUT_CELL_METADATA = 19,
            OUT_CELL_FUNCTION = 39,
            OUT_CELL_FUNCTION_DATA = 40
        };
    };
    struct ScannerOut {
        enum Type {
            SUCCESS,
            FINISHED
        };
    };

    struct Sensor {
        enum Type {
            OUTPUT = 5,
            INPUT = 20,
            INOUT_ANGLE = 21,
            IN_MIN_MASS = 22,
            IN_MAX_MASS = 23,
            OUT_MASS = 24,
            OUT_DISTANCE = 25
        };
    };
    struct SensorIn {
        enum Type {
            DO_NOTHING,
            SEARCH_VICINITY,
            SEARCH_BY_ANGLE,
            SEARCH_FROM_CENTER,
            SEARCH_TOWARD_CENTER,
            _COUNTER
        };
    };
    struct SensorOut {
        enum Type {
            NOTHING_FOUND,
            CLUSTER_FOUND
        };
    };

    struct Digestion {
        enum Type {
            OUTPUT = 5,
        };
    };
    struct DigestionOut {
        enum Type {
            NO_TARGET,
            STRIKE_SUCCESSFUL
        };
    };

    struct Muscle {
        enum Type {
            OUTPUT = 5,
            INPUT = 36,
        };
    };
    struct MuscleOut {
        enum Type {
            SUCCESS,
            LIMIT_REACHED
        };
    };
    struct MuscleIn {
        enum Type {
            DO_NOTHING,
            CONTRACT,
            CONTRACT_RELAX,
            EXPAND,
            EXPAND_RELAX,
            _COUNTER
        };
    };
}

struct InstructionCoded {
    Enums::ComputationOperation operation;
    Enums::ComputationOpType opType1;
    Enums::ComputationOpType opType2;
    uint8_t operand1;
    uint8_t operand2;
};
