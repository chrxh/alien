#include <QPainter>

#include "gui/guisettings.h"
#include "gui/guisettings.h"

#include "cellgraphicsitemconfig.h"
#include "cellgraphicsitem.h"

CellGraphicsItem::CellGraphicsItem (CellGraphicsItemConfig* config, Cell* cell, qreal x, qreal y, bool connectable, int numToken
	, quint8 color, QString displayString, QGraphicsItem *parent)
    : QGraphicsItem(parent), _config(config), _cell(cell), _connectable(connectable), _focusState(NO_FOCUS), _numToken(numToken)
	, _color(color), _displayString(displayString)
{
    QGraphicsItem::setPos(x, y);
}

CellGraphicsItem::~CellGraphicsItem()
{
}

QRectF CellGraphicsItem::boundingRect () const
{
    return QRectF(-1.5*GRAPHICS_ITEM_SIZE, -0.5*GRAPHICS_ITEM_SIZE, 3.0*GRAPHICS_ITEM_SIZE, 1.0*GRAPHICS_ITEM_SIZE);
}

void CellGraphicsItem::paint (QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget)
{
    //set pen color depending on whether the cell is on focus or not
    if( _focusState == NO_FOCUS ) {

        //no pen
        painter->setPen(QPen(QBrush(QColor(0,0,0,0)), 0.03 * GRAPHICS_ITEM_SIZE));
    }
    else {

        //pen
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

	if (_config->showCellFunctions) {
		auto font = GuiFunctions::getCellFont();
		painter->setFont(font);
		painter->setPen(QPen(QBrush(GRAPHICS_ITEM_COLOR), 0.03 * GRAPHICS_ITEM_SIZE));
		painter->drawText(QRectF(-1.5*GRAPHICS_ITEM_SIZE, 0.2*GRAPHICS_ITEM_SIZE, 3.0*GRAPHICS_ITEM_SIZE, 1.0*GRAPHICS_ITEM_SIZE), Qt::AlignCenter, _displayString);
	}

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

}

int CellGraphicsItem::type() const
{
    // enables the use of qgraphicsitem_cast with this item.
    return Type;
}

Cell* CellGraphicsItem::getCell ()
{
    return _cell;
}

void CellGraphicsItem::setConnectable (bool connectable)
{
    _connectable = connectable;
}

CellGraphicsItem::FocusState CellGraphicsItem::getFocusState ()
{
    return _focusState;
}

void CellGraphicsItem::setFocusState (FocusState focusState)
{
    _focusState = focusState;
}

void CellGraphicsItem::setNumToken (int numToken)
{
    _numToken = numToken;
}

void CellGraphicsItem::setColor (quint8 color)
{
    _color = color;
}

void CellGraphicsItem::setDisplayString(QString value)
{
	_displayString = value;
}

