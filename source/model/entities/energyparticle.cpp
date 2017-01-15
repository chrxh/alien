#include <qmath.h>

#include "global/servicelocator.h"
#include "global/global.h"

#include "model/factoryfacade.h"
#include "model/physics/physics.h"
#include "model/config.h"
#include "model/simulationcontext.h"
#include "model/energyparticlemap.h"
#include "model/cellmap.h"
#include "model/topology.h"

#include "energyparticle.h"
#include "grid.h"
#include "cell.h"
#include "cellcluster.h"


EnergyParticle::EnergyParticle(SimulationContext* context)
    : _context(context)
	, _topology(context->getTopology())
	, _cellMap(context->getCellMap())
	, _energyMap(context->getEnergyParticleMap())
	, id(GlobalFunctions::createNewTag())
{

}

EnergyParticle::EnergyParticle(qreal amount_, QVector3D pos_, QVector3D vel_, SimulationContext* context)
    : EnergyParticle(context)
{
	amount = amount_;
	pos = pos_;
	vel = vel_;
}

//return: true = energy is zero
//        cluster is nonzero if particle transforms into cell
bool EnergyParticle::movement (CellCluster*& cluster)
{
    //remove old position from map
    _energyMap->removeParticleIfPresent(pos, this);

    //update position
    pos += vel;
    _topology->correctPosition(pos);

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
    EnergyParticle* otherEnergy(_energyMap->getParticle(pos));
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
        Cell* cell(_cellMap->getCell(pos));
        if( cell ) {
            cell->setEnergy(cell->getEnergy() + amount);
            //create token?
/*            if( (cell->getNumToken(true) < CELL_TOKENSTACKSIZE) && (amount > MIN_TOKEN_ENERGY) ) {
                Token* token = new Token(amount, true);
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
                        if( _cellMap->getCell(pos+QVector3D(dx,dy,0.0)) ) {
                            _energyMap->setParticle(pos, this);
                            return false;
                        }
                    }
                }

                //create cell and cluster
                QList< Cell* > cells;
                FactoryFacade* facade = ServiceLocator::getInstance().getService<FactoryFacade>();
                Cell* c = facade->buildFeaturedCellWithRandomData(eNew, _context);
                cells << c;
                cluster = facade->buildCellCluster(cells, 0.0, pos, 0, vel, _context);
                amount = 0;
                _cellMap->setCell(pos, c);
				CellMetadata meta = c->getMetadata();
				meta.color = color;
                c->setMetadata(meta);
                return true;
            }
            else {
                _energyMap->setParticle(pos, this);
                return false;
            }

        }
    }
}

void EnergyParticle::serializePrimitives (QDataStream& stream)
{
    stream << amount << pos << vel << id << color;
}

void EnergyParticle::deserializePrimitives (QDataStream& stream)
{
    stream >> amount >> pos >> vel >> id >> color;
}





