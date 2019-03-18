#pragma once
#include <QGraphicsItem>

class FastImageItem
	: public QGraphicsItem
{
public:
	FastImageItem(QImage* image);
	~FastImageItem();

	QRectF boundingRect() const override;
	void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget = Q_NULLPTR) override;

private:
	QImage* _image = nullptr;
	
};
