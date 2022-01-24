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
        CellFunction_Computation = 0,
        CellFunction_Propulsion = 1,
        CellFunction_Scanner = 2,
        CellFunction_Digestion = 3,
        CellFunction_Constructor = 4,
        CellFunction_Sensor = 5,
        CellFunction_Muscle = 6,
        CellFunction_Count = 7
    };

    struct EnergyGuidance {
        enum Type {
            INPUT = 1,
            IN_VALUE_CELL = 2,
            IN_VALUE_TOKEN = 3
        };
    };

    struct EnergyGuidanceIn {
        enum Type {
            DEACTIVATED,
            BALANCE_CELL,
            BALANCE_TOKEN,
            BALANCE_BOTH,
            HARVEST_CELL,
            HARVEST_TOKEN,
            _COUNTER
        };
    };

    struct Communicator {
        enum Type {
            INPUT = 26,
            IN_CHANNEL = 27,
            IN_MESSAGE = 28,
            IN_ANGLE = 29,
            IN_DISTANCE = 30,
            OUT_SENT_NUM_MESSAGE = 31,
            OUT_RECEIVED_NEW_MESSAGE = 32,
            OUT_RECEIVED_MESSAGE = 33,
            OUT_RECEIVED_ANGLE = 34,
            OUT_RECEIVED_DISTANCE = 35,
        };
    };
    struct CommunicatorIn {
        enum Type {
            DO_NOTHING,
            SET_LISTENING_CHANNEL,
            SEND_MESSAGE,
            RECEIVE_MESSAGE,
            _COUNTER
        };
    };
    struct CommunicatorOutReceivedNewMessage {
        enum Type {
            NO,
            YES
        };
    };

    struct ComputationOperation {
        enum Type {
            MOV, ADD, SUB, MUL, DIV, XOR, OR, AND, IFG, IFGE, IFE, IFNE, IFLE, IFL, ELSE, ENDIF
        };
    };
    struct ComputationOpType {
        enum Type {
            MEM, MEMMEM, CMEM, CONSTANT
        };
    };

    struct Constr {
        enum Type {
            OUTPUT = 5,
            INPUT = 6,
            IN_OPTION = 7,
            IN_ANGLE_ALIGNMENT = 38,  //0: no alignment, 2: alignment to 180 deg, 3: alignment to 120 deg, ... up to 6
            IN_UNIFORM_DIST = 13,
            INOUT_ANGLE = 15,
            IN_DIST = 16,
            IN_CELL_MAX_CONNECTIONS = 17,              //0: automatically; >0: max connections (not greater than MAX_CELL_CONNECTIONS)
            IN_CELL_BRANCH_NO = 18,
            IN_CELL_METADATA = 19,
            IN_CELL_FUNCTION = 39,
            IN_CELL_FUNCTION_DATA = 40
        };
    };
    struct ConstrOut {
        enum Type {
            SUCCESS,
            ERROR_NO_ENERGY,
            ERROR_CONNECTION,
            ERROR_LOCK,
            ERROR_DIST
        };
    };
    struct ConstrIn {
        enum Type {
            DO_NOTHING,
            CONSTRUCT,
            _COUNTER
        };
    };
    struct ConstrInOption {
        enum Type {
            STANDARD,
            CREATE_EMPTY_TOKEN,
            CREATE_DUP_TOKEN,
            FINISH_NO_SEP,
            FINISH_WITH_SEP,
            FINISH_WITH_EMPTY_TOKEN_SEP,
            FINISH_WITH_DUP_TOKEN_SEP,
            _COUNTER
        };
    };

    struct ConstrInUniformDist {
        enum Type {
            NO,
            YES,
            _COUNTER
        };
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
    Enums::ComputationOperation::Type operation;
    Enums::ComputationOpType::Type opType1;
    Enums::ComputationOpType::Type opType2;
    uint8_t operand1;
    uint8_t operand2;
};
