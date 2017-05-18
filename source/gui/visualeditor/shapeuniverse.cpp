#include <QGraphicsItem>
#include <QGraphicsSceneMouseEvent>
#include <QMatrix4x4>

#include "Base/ServiceLocator.h"
#include "model/Entities/Cell.h"
#include "model/Entities/CellCluster.h"
#include "model/Entities/EnergyParticle.h"
#include "model/Features/CellFunction.h"
#include "model/Context/SimulationParameters.h"
#include "model/Settings.h"
#include "model/BuilderFacade.h"
#include "model/Context/SimulationContext.h"
#include "model/Context/SpaceMetric.h"
#include "model/Context/EnergyParticleMap.h"
#include "gui/Settings.h"

#include "cellgraphicsitem.h"
#include "cellconnectiongraphicsitem.h"
#include "cellgraphicsitemconfig.h"
#include "energygraphicsitem.h"
#include "markergraphicsitem.h"
#include "shapeuniverse.h"

ShapeUniverse::ShapeUniverse(QObject *parent)
	: QGraphicsScene(parent)
{
    setBackgroundBrush(QBrush(UNIVERSE_COLOR));
}

ShapeUniverse::~ShapeUniverse()
{
}




