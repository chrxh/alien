#ifndef SELECTIONMULTIPLYARRANGEMENTDIALOG_H
#define SELECTIONMULTIPLYARRANGEMENTDIALOG_H

#include <QDialog>
#include <QVector3D>

namespace Ui {
class SelectionMultiplyArrangementDialog;
}

class SelectionMultiplyArrangementDialog : public QDialog
{
    Q_OBJECT

public:
    explicit SelectionMultiplyArrangementDialog (QVector3D centerPos, QWidget *parent = 0);
    ~SelectionMultiplyArrangementDialog();

    qreal getInitialPosX ();
    qreal getInitialPosY ();
    bool changeVelocityX ();
    bool changeVelocityY ();
    bool changeAngle();
    bool changeAngularVelocity ();
    qreal getInitialVelX ();
    qreal getInitialVelY ();
    qreal getInitialAngle ();
    qreal getInitialAngVel ();

    int getHorizontalNumber ();
    qreal getHorizontalInterval ();
    qreal getHorizontalVelocityXIncrement ();
    qreal getHorizontalVelocityYIncrement ();
    qreal getHorizontalAngleIncrement ();
    qreal getHorizontalAngularVelocityIncrement ();

    int getVerticalNumber ();
    qreal getVerticalInterval ();
    qreal getVerticalVelocityXIncrement ();
    qreal getVerticalVelocityYIncrement ();
    qreal getVerticalAngleIncrement ();
    qreal getVerticalAngularVelocityIncrement ();


private:
    Ui::SelectionMultiplyArrangementDialog *ui;
};

#endif // SELECTIONMULTIPLYARRANGEMENTDIALOG_H
