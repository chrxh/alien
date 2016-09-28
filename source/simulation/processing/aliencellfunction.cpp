#include "aliencellfunction.h"
#include "../entities/aliencell.h"

#include "../../globaldata/simulationparameters.h"

#include <QString>
#include <QtCore/qmath.h>

AlienCellFunction::AlienCellFunction()
{
}

AlienCellFunction::~AlienCellFunction()
{
}

void AlienCellFunction::runEnergyGuidanceSystem (AlienToken* token, AlienCell* previousCell, AlienCell* cell, AlienGrid*& space)
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
/*    for(qreal i = 0.0; i < 2.0; i = i + 0.123) {
        qreal t = convertDataToShiftLen(convertShiftLenToData(i));
        qDebug("v: %f, n: %f", t, convertDataToShiftLen(convertShiftLenToData(t)));
    }
    for(qreal i = 0.0; i < 360.0; i = i + 3.34) {
        qreal t = convertDataToAngle(convertAngleToData(i));
        qDebug("v: %f, n: %f", t, convertDataToAngle(convertAngleToData(t)));
    }*/
//    qDebug("%d, %d", convertAngleToData(60.468750), convertAngleToData(60.465));
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

qreal AlienCellFunction::convertDataToAngle (quint8 b)
{
    //0 to 127 => 0 to 179 degree
    //128 to 255 => -179 to 0 degree
    if( b < 128 )
        return (0.5+static_cast<qreal>(b))*(180.0/128.0);        //add 0.5 to prevent discretization errors
    else
        return (-256.0-0.5+static_cast<qreal>(b))*(180.0/128.0);
}

quint8 AlienCellFunction::convertAngleToData (qreal a)
{
    //0 to 180 degree => 0 to 128
    //-180 to 0 degree => 128 to 256 (= 0)
    if( a > 180.0 )
        a = a - 360.0;
    if( a <= -180.0 )
        a = a + 360.0;
    int intA = static_cast<int>(a*128.0/180.0);
//    if( intA >= 0 )
    return static_cast<quint8>(intA);
//    else
//        return 127-(int)(a*128.0/180.0);
}

qreal AlienCellFunction::convertDataToShiftLen (quint8 b)
{
//    return simulationParameters.CRIT_CELL_DIST_MIN + (simulationParameters.CRIT_CELL_DIST_MAX-simulationParameters.CRIT_CELL_DIST_MIN)*((qreal)b)/255.0;
    return (0.5+(qreal)b)/100.0;                    //add 0.5 to prevent discretization errors
}

quint8 AlienCellFunction::convertShiftLenToData (qreal len)
{
//    return (quint8)((len - simulationParameters.CRIT_CELL_DIST_MIN)/(simulationParameters.CRIT_CELL_DIST_MAX-simulationParameters.CRIT_CELL_DIST_MIN)*255.0);
    if( static_cast< quint32 >(len*100.0) >= 256 )
        return 255;
    return static_cast< quint8 >(len*100.0);
}

quint8 AlienCellFunction::convertURealToData (qreal r) {
    if( r < 0.0 )
        return 0;
    if( r > 127.0)
        return 127;
    return qFloor(r);
}

/*
qreal AlienCellFunction::convertDataToAngle (quint8 b)
{
    //0-127 => 0-179 degree
    //128-255 => 0-(-179) degree
    if( b < 128 )
        return ((qreal)b)*(180.0/128.0);
    else
        return (127.0-(qreal)b)*(180.0/128.0);
}
*/

