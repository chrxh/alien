#pragma once

#include <QTextEdit>

#include "Gui/Definitions.h"

class HexEditWidget
	: public QTextEdit
{
    Q_OBJECT
public:
    HexEditWidget(QWidget *parent = 0);
    virtual ~HexEditWidget () = default;

    void updateDisplay (QByteArray const& data);
	QByteArray const& getData() const;

    static QByteArray convertHexStringToByteArray (QString hexCode);

	Q_SIGNAL void dataChanged();

protected:
    void keyPressEvent (QKeyEvent* e);
    void mousePressEvent(QMouseEvent* e);
    void mouseDoubleClickEvent (QMouseEvent* e);
    void wheelEvent (QWheelEvent* e);

	Q_SIGNAL void cursorReachedBeginning (int newCol);   //newCol = -1: end of previous block
	Q_SIGNAL void cursorReachedEnd (int newCol);

private:
	QByteArray _data;
};
