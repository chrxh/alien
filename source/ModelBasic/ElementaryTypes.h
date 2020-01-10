#pragma once

namespace Enums
{
    struct Branching {
        enum Type {
            TOKEN_BRANCH_NUMBER = 0
        };
    };

    struct CellFunction {
        enum Type {
            COMPUTER,
            PROPULSION,
            SCANNER,
            WEAPON,
            CONSTRUCTOR,
            SENSOR,
            COMMUNICATOR,
            _COUNTER
        };
    };

    struct EnergyGuidance {
        enum Type {
            IN = 1,
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
            IN = 26,
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

    struct ComputerOperation {
        enum Type {
            MOV, ADD, SUB, MUL, DIV, XOR, OR, AND, IFG, IFGE, IFE, IFNE, IFLE, IFL, ELSE, ENDIF
        };
    };
    struct ComputerOptype {
        enum Type {
            MEM, MEMMEM, CMEM, CONST
        };
    };

    struct Constr {
        enum Type {
            OUT = 5,
            IN = 6,
            IN_OPTION = 7,
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
            SUCCESS_ROT,
            ERROR_NO_ENERGY,
            ERROR_OBSTACLE,
            ERROR_CONNECTION,
            ERROR_DIST,
            ERROR_MAX_RADIUS
        };
    };
    struct ConstrIn {
        enum Type {
            DO_NOTHING,
            SAFE,
            UNSAFE,
            BRUTEFORCE,
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
            FINISH_WITH_SEP_RED,
            FINISH_WITH_TOKEN_SEP_RED,
            _COUNTER
        };
    };

    struct Prop {
        enum Type {
            OUT = 5,
            IN = 8,
            IN_ANGLE = 9,
            IN_POWER = 10
        };
    };
    struct PropOut {
        enum Type {
            SUCCESS,
            SUCCESS_DAMPING_FINISHED,
            ERROR_NO_ENERGY
        };
    };
    struct PropIn {
        enum Type {
            DO_NOTHING,
            BY_ANGLE,
            FROM_CENTER,
            TOWARD_CENTER,
            ROTATION_CLOCKWISE,
            ROTATION_COUNTERCLOCKWISE,
            DAMP_ROTATION,
            _COUNTER
        };
    };

    struct Scanner {
        enum Type {
            OUT = 5,
            INOUT_CELL_NUMBER = 12,
            OUT_MASS = 13,
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
            FINISHED,
            RESTART
        };
    };

    struct Sensor {
        enum Type {
            OUT = 5,
            IN = 20,
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

    struct Weapon {
        enum Type {
            OUT = 5,
            IN_MIN_MASS = 26,
            IN_MAX_MASS = 27,
        };
    };
    struct WeaponOut {
        enum Type {
            NO_TARGET,
            STRIKE_SUCCESSFUL
        };
    };
}

struct InstructionCoded {
    Enums::ComputerOperation::Type operation;
    Enums::ComputerOptype::Type opType1;
    Enums::ComputerOptype::Type opType2;
    uint8_t operand1;
    uint8_t operand2;
};
