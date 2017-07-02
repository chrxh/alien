#ifndef ENERGYEDIT_H
#define ENERGYEDIT_H

#include <QTextEdit>
#include <QVector2D>

#include "Model/Entities/CellTO.h"

class Cell;
class EnergyEdit : public QTextEdit
{
    Q_OBJECT
public:
    explicit EnergyEdit(QWidget *parent = 0);

    void updateEnergyParticle (QVector2D pos, QVector2D vel, qreal energy);
    void requestUpdate ();

Q_SIGNALS:
    void energyParticleDataChanged (QVector2D pos, QVector2D vel, qreal energyValue);

protected:
    void keyPressEvent (QKeyEvent* e);
    void mousePressEvent(QMouseEvent* e);
    void mouseDoubleClickEvent (QMouseEvent* e);
    void wheelEvent (QWheelEvent* e);

private:

    void updateDisplay();

    qreal generateNumberFromFormattedString (QString s);
    QString generateFormattedRealString (QString s);
    QString generateFormattedRealString (qreal r);

    QVector2D _energyParticlePos;
    QVector2D _energyParticleVel;
    qreal _energyParticleValue;
};

#endif // ENERGYEDIT_H
