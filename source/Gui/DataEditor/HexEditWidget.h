#pragma once

#include <QTextEdit>

class HexEditWidget : public QTextEdit
{
    Q_OBJECT
public:
    HexEditWidget(QWidget *parent = 0);
    ~HexEditWidget ();

    void update ();
    void update (QByteArray const& data);

	QByteArray const& getDataRef ();

    static QByteArray convertHexStringToByteArray (QString hexCode);

protected:
    void keyPressEvent (QKeyEvent* e);
    void mousePressEvent(QMouseEvent* e);
    void mouseDoubleClickEvent (QMouseEvent* e);
    void wheelEvent (QWheelEvent* e);

Q_SIGNALS:
    void dataChanged (QByteArray data);
    void cursorReachedBeginning (int newCol);   //newCol = -1: end of previous block
    void cursorReachedEnd (int newCol);

private:
    void displayData ();

    QByteArray _data;
};
