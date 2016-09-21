#ifndef ALIENCELLFUNCTION_H
#define ALIENCELLFUNCTION_H

#include "../entities/alientoken.h"

class AlienCellCluster;
class AlienGrid;
class AlienEnergy;
class AlienCellFunction
{
public:
    AlienCellFunction();
    virtual ~AlienCellFunction();

    virtual void runEnergyGuidanceSystem (AlienToken* token, AlienCell* previousCell, AlienCell* cell, AlienGrid*& space);
    virtual void execute (AlienToken* token, AlienCell* previousCell, AlienCell* cell, AlienGrid*& space, AlienEnergy*& newParticle, bool& decompose) = 0;
    virtual QString getCode ();
    virtual bool compileCode (QString code, int& errorLine);
    virtual QString getCellFunctionName () = 0;

    virtual void serialize (QDataStream& stream);

    virtual void getInternalData (quint8* data);

protected:
    qreal convertDataToAngle (quint8 b);
    quint8 convertAngleToData (qreal a);
    qreal convertDataToShiftLen (quint8 b);
    quint8 convertShiftLenToData (qreal len);
    quint8 convertURealToData (qreal r);

    //*******************************************
    //* constants for cell function programming *
    //*******************************************
    //------- ENERGY GUIDANCE-----------
    enum class ENERGY_GUIDANCE {
        IN = 1,
        IN_VALUE_CELL = 2,
        IN_VALUE_TOKEN = 3
    };
    enum class ENERGY_GUIDANCE_IN {
        DEACTIVATED,
        BALANCE_CELL,
        BALANCE_TOKEN,
        BALANCE_BOTH,
        HARVEST_CELL,
        HARVEST_TOKEN
    };

    //------- COMPUTER -----------
    enum class COMPUTER_OPERATION {
        MOV, ADD, SUB, MUL, DIV, XOR, OR, AND, IFG, IFGE, IFE, IFNE, IFLE, IFL, ELSE, ENDIF
    };
    enum class COMPUTER_OPTYPE {
        MEM, MEMMEM, CMEM, CONST
    };

    //------- CONSTRUCTOR -----------
    enum class CONSTR {
        OUT = 5,
        IN = 6,
        IN_OPTION = 7,
        INOUT_ANGLE = 15,
        IN_DIST = 16,
        IN_CELL_MAX_CONNECTIONS = 17,              //0: automatically; >0: max connections (not greater than MAX_CELL_CONNECTIONS)
        IN_CELL_BRANCH_NO = 18,
        IN_CELL_FUNCTION = 19,
        IN_CELL_FUNCTION_DATA = 30
    };
    enum class CONSTR_OUT {
        SUCCESS,
        SUCCESS_ROT,
        ERROR_NO_ENERGY,
        ERROR_OBSTACLE,
        ERROR_CONNECTION,
        ERROR_DIST
    };
    enum class CONSTR_IN {
        DO_NOTHING,
        SAFE,
        UNSAFE,
        BRUTEFORCE
    };
    enum class CONSTR_IN_OPTION {
        STANDARD,
        CREATE_EMPTY_TOKEN,
        CREATE_DUP_TOKEN,
        FINISH_NO_SEP,
        FINISH_WITH_SEP,
        FINISH_WITH_SEP_RED,
        FINISH_TOKEN_WITH_SEP_RED
    };
    enum class CONSTR_IN_CELL_TYPE {
        COMPUTER,
        PROP,
        SCANNER,
        WEAPON,
        CONSTR,
        SENSOR
    };

    //------- PROPULSION -----------
    enum class PROP {
        OUT = 5,
        IN = 8,
        IN_ANGLE = 9,
        IN_POWER = 10
    };
    enum class PROP_OUT {
        SUCCESS,
        SUCCESS_DAMPING_FINISHED,
        ERROR_NO_ENERGY
    };
    enum class PROP_IN {
        DO_NOTHING,
        BY_ANGLE,
        FROM_CENTER,
        TOWARD_CENTER,
        ROTATION_CLOCKWISE,
        ROTATION_COUNTERCLOCKWISE,
        DAMP_ROTATION
    };

    //------- SCANNER -----------
    enum class SCANNER {
        OUT = 5,
//        IN = 11,
        INOUT_CELL_NUMBER = 12,
        OUT_MASS = 13,
        OUT_ENERGY = 14,
        OUT_ANGLE = 15,
        OUT_DIST = 16,
        OUT_CELL_MAX_CONNECTIONS = 17,
        OUT_CELL_BRANCH_NO = 18,
        OUT_CELL_FUNCTION = 19,
        OUT_CELL_FUNCTION_DATA = 30
    };
    enum class SCANNER_OUT {
        SUCCESS,
        FINISHED,
        RESTART
    };
    enum class SCANNER_OUT_CELL_TYPE {
        COMPUTER,
        PROP,
        SCANNER,
        WEAPON,
        CONSTR,
        SENSOR
    };

    //------- WEAPON -----------
    enum class WEAPON {
        OUT = 5,
    };
    enum class WEAPON_OUT {
        NO_TARGET,
        STRIKE_SUCCESSFUL
    };

    //------- SENSOR -----------
    enum class SENSOR {
        OUT = 5,
        IN = 20,
        INOUT_ANGLE = 21,
        IN_MIN_MASS = 22,
        IN_MAX_MASS = 23,
        OUT_MASS = 24,
        OUT_DIST = 25
    };
    enum class SENSOR_IN {
        DO_NOTHING,
        SEARCH_VICINITY,
        SEARCH_BY_ANGLE,
        SEARCH_FROM_CENTER,
        SEARCH_TOWARD_CENTER
    };
    enum class SENSOR_OUT {
        NOTHING_FOUND,
        CLUSTER_FOUND
    };

};

#endif // ALIENCELLFUNCTION_H
