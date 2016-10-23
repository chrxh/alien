#ifndef ENERGYEDIT_H
#define ENERGYEDIT_H

#include <QTextEdit>
#include <QVector3D>

#include "model/entities/aliencellto.h"

class AlienCell;
class EnergyEdit : public QTextEdit
{
    Q_OBJECT
public:
    explicit EnergyEdit(QWidget *parent = 0);

    void updateEnergyParticle (QVector3D pos, QVector3D vel, qreal energy);
    void requestUpdate ();

signals:
    void energyParticleDataChanged (QVector3D pos, QVector3D vel, qreal energyValue);

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

    QVector3D _energyParticlePos;
    QVector3D _energyParticleVel;
    qreal _energyParticleValue;
};

#endif // ENERGYEDIT_H
