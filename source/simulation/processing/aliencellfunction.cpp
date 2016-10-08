#include "aliencellfunction.h"
#include "../entities/aliencell.h"
#include "../entities/aliengrid.h"
#include "../physics/physics.h"

#include "../../globaldata/simulationsettings.h"

#include <QString>
#include <QtCore/qmath.h>

AlienCellFunction::AlienCellFunction()
{
}

AlienCellFunction::~AlienCellFunction()
{
}

void AlienCellFunction::runEnergyGuidanceSystem (AlienToken* token, AlienCell* cell, AlienCell* previousCell, AlienGrid*& space)
{
    quint8 cmd = token->memory[static_cast<int>(ENERGY_GUIDANCE::IN)] % 6;
    qreal valueCell = token->memory[static_cast<int>(ENERGY_GUIDANCE::IN_VALUE_CELL)];
    qreal valueToken = token->memory[static_cast<int>(ENERGY_GUIDANCE::IN_VALUE_TOKEN)];
    qreal amount = 10.0;

    if( cmd == static_cast<int>(ENERGY_GUIDANCE_IN::BALANCE_CELL) ) {
        if( cell->getEnergy() > (simulationParameters.CRIT_CELL_TRANSFORM_ENERGY+valueCell) ) {
            cell->setEnergy(cell->getEnergy()-amount);
            token->energy += amount;
        }
        if( cell->getEnergy() < (simulationParameters.CRIT_CELL_TRANSFORM_ENERGY+valueCell) ) {
            if( token->energy > (simulationParameters.MIN_TOKEN_ENERGY+valueToken+amount) ) {
                cell->setEnergy(cell->getEnergy()+amount);
                token->energy -= amount;
            }
        }
    }
    if( cmd == static_cast<int>(ENERGY_GUIDANCE_IN::BALANCE_TOKEN) ) {
        if( token->energy > (simulationParameters.MIN_TOKEN_ENERGY+valueToken) ) {
            cell->setEnergy(cell->getEnergy()+amount);
            token->energy -= amount;
        }
        if( token->energy < (simulationParameters.MIN_TOKEN_ENERGY+valueToken) ) {
            if( cell->getEnergy() > (simulationParameters.CRIT_CELL_TRANSFORM_ENERGY+valueCell+amount) ) {
                cell->setEnergy(cell->getEnergy()-amount);
                token->energy += amount;
            }
        }
    }
    if( cmd == static_cast<int>(ENERGY_GUIDANCE_IN::BALANCE_BOTH) ) {
        if( (token->energy > (simulationParameters.MIN_TOKEN_ENERGY+valueToken+amount))
                && (cell->getEnergy() < (simulationParameters.CRIT_CELL_TRANSFORM_ENERGY+valueCell)) ) {
            cell->setEnergy(cell->getEnergy()+amount);
            token->energy -= amount;
        }
        if( (token->energy < (simulationParameters.MIN_TOKEN_ENERGY+valueToken))
                && (cell->getEnergy() > (simulationParameters.CRIT_CELL_TRANSFORM_ENERGY+valueCell+amount)) ) {
            cell->setEnergy(cell->getEnergy()-amount);
            token->energy += amount;
        }
    }
    if( cmd == static_cast<int>(ENERGY_GUIDANCE_IN::HARVEST_CELL) ) {
        if( cell->getEnergy() > (simulationParameters.CRIT_CELL_TRANSFORM_ENERGY+valueCell+amount) ) {
            cell->setEnergy(cell->getEnergy()-amount);
            token->energy += amount;
        }
    }
    if( cmd == static_cast<int>(ENERGY_GUIDANCE_IN::HARVEST_TOKEN) ) {
        if( token->energy > (simulationParameters.MIN_TOKEN_ENERGY+valueToken+amount) ) {
            cell->setEnergy(cell->getEnergy()+amount);
            token->energy -= amount;
        }
    }
}

QString AlienCellFunction::getCode ()
{
    return QString();
}

bool AlienCellFunction::compileCode (QString code, int& errorLine)
{
    return true;
}

void AlienCellFunction::serialize (QDataStream& stream)
{
    stream << getCellFunctionName();
}

void AlienCellFunction::getInternalData (quint8* data)
{

}

qreal AlienCellFunction::calcAngle (AlienCell* origin, AlienCell* ref1, AlienCell* ref2, AlienGrid* grid) const
{
    qreal a1 = Physics::calcAngle(ref1->calcPosition() - origin->calcPosition());
    qreal a2 = Physics::calcAngle(-ref2->calcPosition() + origin->calcPosition());
    qreal angle = a1 - a2;
    return angle;   //not correct in every case
}

qreal AlienCellFunction::convertDataToAngle (quint8 b) const
{
    //0 to 127 => 0 to 179 degree
    //128 to 255 => -179 to 0 degree
    if( b < 128 )
        return (0.5+static_cast<qreal>(b))*(180.0/128.0);        //add 0.5 to prevent discretization errors
    else
        return (-256.0-0.5+static_cast<qreal>(b))*(180.0/128.0);
}

quint8 AlienCellFunction::convertAngleToData (qreal a) const
{
    //0 to 180 degree => 0 to 128
    //-180 to 0 degree => 128 to 256 (= 0)
    if( a > 180.0 )
        a = a - 360.0;
    if( a <= -180.0 )
        a = a + 360.0;
    int intA = static_cast<int>(a*128.0/180.0);
    return static_cast<quint8>(intA);
}

qreal AlienCellFunction::convertDataToShiftLen (quint8 b) const
{
    return (0.5+(qreal)b)/100.0;                    //add 0.5 to prevent discretization errors
}

quint8 AlienCellFunction::convertShiftLenToData (qreal len) const
{
    if( static_cast< quint32 >(len*100.0) >= 256 )
        return 255;
    return static_cast< quint8 >(len*100.0);
}

quint8 AlienCellFunction::convertURealToData (qreal r) const
{
    if( r < 0.0 )
        return 0;
    if( r > 127.0)
        return 127;
    return qFloor(r);
}

quint8 AlienCellFunction::convertIntToData (int i) const
{
    if( i > 127 )
        return i;
    if( i < -128)
        return -128;
    return i;
}

