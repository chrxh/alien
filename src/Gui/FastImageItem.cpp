#include <QPainter>

#include "FastImageItem.h"

FastImageItem::FastImageItem(QImage* image)
	: QGraphicsItem(), _image(image)
{
	
}

FastImageItem::~FastImageItem() {
	
}


QRectF FastImageItem::boundingRect() const
{
	return QRectF(0, 0, _image->width(), _image->height());
}

void FastImageItem::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget /*= Q_NULLPTR*/)
{
	painter->drawImage(0, 0, *_image);
}

