#include <QPainter>

#include "Model/Entities/Descriptions.h"

#include "Gui/Settings.h"

#include "ItemConfig.h"
#include "CellItem.h"

CellItem::CellItem (ItemConfig* config, CellDescription const& desc, QGraphicsItem* parent /*= nullptr*/)
    : AbstractItem(parent), _config(config)
{
	update(desc);
}

void CellItem::update(CellDescription const & desc)
{
	auto pos = desc.pos.getValueOrDefault();
	QGraphicsItem::setPos(QPointF(pos.x() * GRAPHICS_ITEM_SIZE, pos.y() * GRAPHICS_ITEM_SIZE));

	_numToken = desc.tokens.getValueOrDefault().size();
	_color = desc.metadata.getValueOrDefault().color;
	_branchNumber = desc.tokenBranchNumber.getValueOr(0);
	auto numConnections = desc.connectingCells.getValueOrDefault().size();
	auto maxConnections = desc.maxConnections.getValueOr(0);
	_connectable = (numConnections < maxConnections);
	_id = desc.id;
}

QRectF CellItem::boundingRect () const
{
    return QRectF(-0.5*GRAPHICS_ITEM_SIZE, -0.5*GRAPHICS_ITEM_SIZE, 1.0*GRAPHICS_ITEM_SIZE, 1.0*GRAPHICS_ITEM_SIZE);
}

void CellItem::paint (QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget)
{
    //set pen color depending on whether the cell is on focus or not
    if( _focusState == NO_FOCUS ) {
        painter->setPen(QPen(QBrush(QColor(0,0,0,0)), 0.03 * GRAPHICS_ITEM_SIZE));
    }
    else {
        painter->setPen(QPen(QBrush(CELL_CLUSTER_PEN_FOCUS_COLOR), 0.03 * GRAPHICS_ITEM_SIZE));
    }

    //set brush color
	QColor brushColor;
    if( _color == 0 )
       brushColor = INDIVIDUAL_CELL_COLOR1;
    if( _color == 1 )
        brushColor = INDIVIDUAL_CELL_COLOR2;
    if( _color == 2 )
        brushColor = INDIVIDUAL_CELL_COLOR3;
    if( _color == 3 )
        brushColor = INDIVIDUAL_CELL_COLOR4;
    if( _color == 4 )
        brushColor = INDIVIDUAL_CELL_COLOR5;
    if( _color == 5 )
        brushColor = INDIVIDUAL_CELL_COLOR6;
    if( _color >= 6 )
        brushColor = INDIVIDUAL_CELL_COLOR7;
    if( !_connectable )
        brushColor.setHsl(brushColor.hslHue(), brushColor.hslSaturation(), qMax(0, brushColor.lightness()-60), brushColor.alpha());
    painter->setBrush(QBrush(brushColor));

    //draw cell
    if( (_focusState == NO_FOCUS) || (_focusState == FOCUS_CLUSTER) )
        painter->drawEllipse(QPointF(0.0, 0.0), 0.33*GRAPHICS_ITEM_SIZE, 0.33*GRAPHICS_ITEM_SIZE);
    else
        painter->drawEllipse(QPointF(0.0, 0.0), 0.5*GRAPHICS_ITEM_SIZE, 0.5*GRAPHICS_ITEM_SIZE);

    //draw token
    if( _numToken > 0 ) {
        if( _focusState == NO_FOCUS )
            painter->setBrush(QBrush(TOKEN_COLOR));
        else
            painter->setBrush(QBrush(TOKEN_FOCUS_COLOR));
        painter->setPen(QPen(QBrush(CELL_CLUSTER_PEN_FOCUS_COLOR), 0.03 * GRAPHICS_ITEM_SIZE));
        qreal shift1 = -0.5*0.20*(qreal)(_numToken-1);
        if( _numToken > 3)
            shift1 = -0.5*0.20*2.0;
        qreal shiftY1 = -0.5*0.35*(qreal)((_numToken-1)/3);
        for( int i = 0; i < _numToken; ++i) {
            qreal shift2 = 0.20*(qreal)(i%3);
            qreal shiftY2 = 0.35*(qreal)(i/3);
            if( _numToken <= 3 )
                painter->drawEllipse(QPointF(shift1+shift2, shift1+shift2+shiftY1+shiftY2), 0.2*GRAPHICS_ITEM_SIZE, 0.2*GRAPHICS_ITEM_SIZE);
            else
                painter->drawEllipse(QPointF(shift1+shift2, shift1+shift2+shiftY1+shiftY2), 0.1*GRAPHICS_ITEM_SIZE, 0.1*GRAPHICS_ITEM_SIZE);
        }
    }

	if (_config->isShowCellInfo()) {
		auto font = GuiSettings::getCellFont();
		painter->setFont(font);
		painter->setPen(QPen(QBrush(CELLFUNCTION_INFO_COLOR), 0.03 * GRAPHICS_ITEM_SIZE));
		painter->drawText(QRectF(-1.5*GRAPHICS_ITEM_SIZE, 0.1*GRAPHICS_ITEM_SIZE, 3.0*GRAPHICS_ITEM_SIZE, 1.0*GRAPHICS_ITEM_SIZE), Qt::AlignCenter, _displayString);
		painter->setPen(QPen(QBrush(BRANCHNUMBER_INFO_COLOR), 0.03 * GRAPHICS_ITEM_SIZE));
		painter->drawText(QRectF(-0.49*GRAPHICS_ITEM_SIZE, -0.47*GRAPHICS_ITEM_SIZE, 1.0*GRAPHICS_ITEM_SIZE, 1.0*GRAPHICS_ITEM_SIZE), Qt::AlignCenter, QString::number(_branchNumber));
	}

}

int CellItem::type() const
{
    // enables the use of qgraphicsitem_cast with this item.
    return Type;
}

uint64_t CellItem::getId() const
{
	return _id;
}

std::vector<uint64_t> CellItem::getConnectedIds() const
{
	return std::vector<uint64_t>();
}

void CellItem::setConnectable (bool connectable)
{
    _connectable = connectable;
}

CellItem::FocusState CellItem::getFocusState ()
{
    return _focusState;
}

void CellItem::setFocusState (FocusState focusState)
{
    _focusState = focusState;
}

void CellItem::setNumToken (int numToken)
{
    _numToken = numToken;
}

void CellItem::setColor (quint8 color)
{
    _color = color % CELL_COLOR_COUNT;
}

void CellItem::setDisplayString(QString value)
{
	_displayString = value;
}

void CellItem::setBranchNumber(int value)
{
	_branchNumber = value;
}

