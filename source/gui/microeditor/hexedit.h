#ifndef HEXEDIT_H
#define HEXEDIT_H

#include <QTextEdit>

class HexEdit : public QTextEdit
{
    Q_OBJECT
public:
    HexEdit(QWidget *parent = 0);
    ~HexEdit ();

    void update ();
    void update (QByteArray const& data);

	QByteArray const& getDataRef ();

    static QByteArray convertHexStringToByteArray (QString hexCode);

protected:
    void keyPressEvent (QKeyEvent* e);
    void mousePressEvent(QMouseEvent* e);
    void mouseDoubleClickEvent (QMouseEvent* e);
    void wheelEvent (QWheelEvent* e);

signals:
    void dataChanged (QByteArray data);
    void cursorReachedBeginning (int newCol);   //newCol = -1: end of previous block
    void cursorReachedEnd (int newCol);

private:
    void displayData ();

    QByteArray _data;
};

#endif // HEXEDIT_H
