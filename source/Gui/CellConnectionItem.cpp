#include <QPainter>
#include <qmath.h>

#include "ModelBasic/ChangeDescriptions.h"
#include "Gui/Settings.h"

#include "ItemConfig.h"
#include "CellConnectionItem.h"
#include "CoordinateSystem.h"

CellConnectionItem::CellConnectionItem(ItemConfig* config, CellDescription const & cell1, CellDescription const & cell2, QGraphicsItem * parent)
	: AbstractItem(parent), _config(config)
{
	QGraphicsItem::setZValue(-1.0);
	update(cell1, cell2);
}

void CellConnectionItem::update(CellDescription const & cell1, CellDescription const & cell2)
{
	auto pos1 = CoordinateSystem::modelToScene(*cell1.pos);
	auto pos2 = CoordinateSystem::modelToScene(*cell2.pos);
	_dx = (pos2.x() - pos1.x());
	_dy = (pos2.y() - pos1.y());

	QGraphicsItem::setPos(QPointF(pos1.x(), pos1.y()));

	auto branchNumber1 = cell1.tokenBranchNumber.get_value_or(0);
	auto branchNumber2 = cell2.tokenBranchNumber.get_value_or(0);
	auto maxBranchNumber = _config->getSimulationParameters().cellMaxTokenBranchNumber;
	if (branchNumber1 == (branchNumber2 + 1) % maxBranchNumber) {
		_connectionState = ConnectionState::B_TO_A_CONNECTION;
	}
	else if (branchNumber2 == (branchNumber1 + 1) % maxBranchNumber) {
		_connectionState = ConnectionState::A_TO_B_CONNECTION;
	}
	else {
		_connectionState = ConnectionState::NO_DIR_CONNECTION;
	}
}

QRectF CellConnectionItem::boundingRect () const
{
    qreal minX = qMin(0.0, _dx) - CoordinateSystem::modelToScene(0.05);
    qreal minY = qMin(0.0, _dy) - CoordinateSystem::modelToScene(0.05);
    qreal maxX = qMax(0.0, _dx) + CoordinateSystem::modelToScene(0.05);
    qreal maxY = qMax(0.0, _dy) + CoordinateSystem::modelToScene(0.05);
    return QRectF(minX, minY, (maxX-minX), (maxY-minY));
}

void CellConnectionItem::paint (QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget)
{
    if( _connectionState == NO_DIR_CONNECTION )
        painter->setPen(QPen(QBrush(Const::LineInactiveColor), CoordinateSystem::modelToScene(0.06)));
    else
        painter->setPen(QPen(QBrush(Const::LineActiveColor), CoordinateSystem::modelToScene(0.06)));
    painter->drawLine(QPointF(0.0, 0.0), QPointF(_dx, _dy));

    if( (_connectionState == A_TO_B_CONNECTION) || (_connectionState == B_TO_A_CONNECTION) ) {
        qreal x2 = _dx;
        qreal y2 = _dy;
        qreal relPosX = -_dx;
        qreal relPosY = -_dy;
        if( _connectionState == B_TO_A_CONNECTION ) {
            x2 = 0.0;
            y2 = 0.0;
            relPosX = _dx;
            relPosY = _dy;
        }

        qreal len = qSqrt(relPosX*relPosX+relPosY*relPosY);
        relPosX = CoordinateSystem::modelToScene(relPosX / len);
        relPosY = CoordinateSystem::modelToScene(relPosY / len);

        //rotate 45 degree counterclockwise and scaling
        qreal aX = (relPosX-relPosY)/10.0;
        qreal aY = (relPosX+relPosY)/10.0;
        qreal bX = x2 + relPosX*0.35;
        qreal bY = y2 + relPosY*0.35;
        painter->drawLine(QPointF(bX, bY), QPointF(bX+aX, bY+aY));

        //rotate 45 degree clockwise
        aX = (relPosX+relPosY)/10.0;
        aY = (-relPosX+relPosY)/10.0;
        painter->drawLine(QPointF(bX, bY), QPointF(bX+aX, bY+aY));
    }
}

void CellConnectionItem::setConnectionState (ConnectionState connectionState)
{
    _connectionState = connectionState;
}
