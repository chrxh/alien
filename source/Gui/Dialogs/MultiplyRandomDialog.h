#pragma once

#include <QDialog>

namespace Ui {
class MultiplyRandomDialog;
}

class MultiplyRandomDialog : public QDialog
{
    Q_OBJECT

public:
    explicit MultiplyRandomDialog(QWidget *parent = nullptr);
    virtual ~MultiplyRandomDialog();

    int getNumber ();
    bool randomizeAngle ();
    qreal randomizeAngleMin ();
    qreal randomizeAngleMax ();
    bool randomizeVelX ();
    qreal randomizeVelXMin ();
    qreal randomizeVelXMax ();
    bool randomizeVelY ();
    qreal randomizeVelYMin ();
    qreal randomizeVelYMax ();
    bool randomizeAngVel ();
    qreal randomizeAngVelMin ();
    qreal randomizeAngVelMax ();

private:
    Ui::MultiplyRandomDialog *ui;
};
