#include "alienenergy.h"
#include "aliengrid.h"
#include "aliencell.h"
#include "aliencellcluster.h"
#include "../physics/physics.h"

#include "global/globalfunctions.h"
#include "model/simulationsettings.h"
#include <qmath.h>

AlienEnergy::AlienEnergy(qreal amount_, QVector3D pos_, QVector3D vel_, AlienGrid*& grid)
    : _grid(grid), amount(amount_), pos(pos_), vel(vel_), id(GlobalFunctions::getTag()), color(0)
{
}

AlienEnergy::AlienEnergy (QDataStream& stream, QMap< quint64, AlienEnergy* >& oldIdEnergyMap, AlienGrid*& grid)
    : _grid(grid)
{
    stream >> amount >> pos >> vel >> id >> color;
    oldIdEnergyMap[id] = this;
}

//return: true = energy is zero
//        cluster is nonzero if particle transforms into cell
bool AlienEnergy::movement (AlienCellCluster*& cluster)
{
    //remove old position from map
    _grid->removeEnergy(pos, this);

    //update position
    pos += vel;
    _grid->correctPosition(pos);

    //apply gravitational force
/*    QVector3D gSource1(200.0+qSin(0.5*degToRad*(qreal)time)*50, 200.0+qCos(0.5*degToRad*(qreal)time)*50, 0.0);
    QVector3D gSource2(200.0-qSin(0.5*degToRad*(qreal)time)*50, 200.0-qCos(0.5*degToRad*(qreal)time)*50, 0.0);
    QVector3D distance1 = gSource1-pos;
    QVector3D distance2 = gSource1-(pos+vel);
    grid->correctDistance(distance1);
    grid->correctDistance(distance2);
    vel += (distance1.normalized()/(distance1.lengthSquared()+4.0));
    vel += (distance2.normalized()/(distance2.lengthSquared()+4.0));
    distance1 = gSource2-pos;
    distance2 = gSource2-(pos+vel);
    grid->correctDistance(distance1);
    grid->correctDistance(distance2);
    vel += (distance1.normalized()/(distance1.lengthSquared()+4.0));
    vel += (distance2.normalized()/(distance2.lengthSquared()+4.0));
*/
    //is there energy at new position?
    AlienEnergy* otherEnergy(_grid->getEnergy(pos));
    if( otherEnergy ) {

        //particle with most energy inherits color
        if( otherEnergy->amount < amount)
            otherEnergy->color = color;

        otherEnergy->amount += amount;
        amount = 0;
        otherEnergy->vel = (otherEnergy->vel + vel)/2.0;

        return true;
    }
    else {
        //is there a cell at new position?
        AlienCell* cell(_grid->getCell(pos));
        if( cell ) {
            cell->setEnergy(cell->getEnergy() + amount);
            //create token?
/*            if( (cell->getNumToken(true) < CELL_TOKENSTACKSIZE) && (amount > MIN_TOKEN_ENERGY) ) {
                AlienToken* token = new AlienToken(amount, true);
                cell->addToken(token,false);
            }
            else
                cell->setEnergy(cell->getEnergy() + amount);
*/
            amount = 0;
            return true;
        }
        else {

            //enough energy for cell transformation?
            qreal p((qreal)qrand()/RAND_MAX);
            qreal eKin = Physics::kineticEnergy(1, vel, 0, 0);
            qreal eNew = amount - (eKin/simulationParameters.INTERNAL_TO_KINETIC_ENERGY);
            if( (eNew >= simulationParameters.CRIT_CELL_TRANSFORM_ENERGY) && ( p < simulationParameters.CELL_TRANSFORM_PROB) ) {

                //look for neighbor cell
                for(int dx = -2; dx < 3; ++dx ) {
                    for(int dy = -2; dy < 3; ++dy ) {
                        if( _grid->getCell(pos+QVector3D(dx,dy,0.0)) ) {
                            _grid->setEnergy(pos, this);
                            return false;
                        }
                    }
                }

                //create cell and cluster
                QList< AlienCell* > cells;
                AlienCell* c = AlienCell::buildCellWithRandomData(eNew, _grid);
                cells << c;
                cluster = AlienCellCluster::buildCellCluster(cells, 0.0, pos, 0, vel, _grid);
                amount = 0;
                _grid->setCell(pos, c);
                c->setColor(color);
                return true;
            }
            else {
                _grid->setEnergy(pos, this);
                return false;
            }

        }
    }
}

void AlienEnergy::serialize (QDataStream& stream)
{
    stream << amount << pos << vel << id << color;
}






