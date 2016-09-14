#ifndef SELECTIONMULTIPLYRANDOMDIALOG_H
#define SELECTIONMULTIPLYRANDOMDIALOG_H

#include <QDialog>

namespace Ui {
class SelectionMultiplyRandomDialog;
}

class SelectionMultiplyRandomDialog : public QDialog
{
    Q_OBJECT

public:
    explicit SelectionMultiplyRandomDialog(QWidget *parent = 0);
    ~SelectionMultiplyRandomDialog();

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
    Ui::SelectionMultiplyRandomDialog *ui;
};

#endif // SELECTIONMULTIPLYRANDOMDIALOG_H
